//! Physical placement of shard extents and adjacency coalescing.
//!
//! The manifest is sorted by shard id, which has no relation to the byte layout.
//! A positional reader wants the offsets in ascending order, and it wants
//! adjacent extents merged into one read. This module owns that mapping. It
//! never changes which shards exist; it only changes the order and grouping of
//! the reads that fetch them.

use std::collections::BTreeMap;

use selvedge_core::ShardId;
use selvedge_format::Manifest;
use serde::Serialize;

use crate::error::{Result, StoreError};

/// One shard extent inside the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Extent {
    /// Shard.
    pub shard: ShardId,
    /// Payload-relative offset.
    pub offset: u64,
    /// Extent length.
    pub len: u64,
}

impl Extent {
    /// Exclusive end offset.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.len)
    }
}

/// One read that covers several adjacent shards.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    /// Payload-relative start offset.
    pub offset: u64,
    /// Total byte length.
    pub length: u64,
    /// Shards covered, with their offset inside the run.
    pub parts: Vec<RunPart>,
}

/// One shard inside a [`Run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunPart {
    /// Shard.
    pub shard: ShardId,
    /// Offset inside the run buffer.
    pub offset: u64,
    /// Shard length.
    pub len: u64,
}

/// Physical layout of a validated manifest.
#[derive(Clone, Debug)]
pub struct Layout {
    by_shard: BTreeMap<ShardId, Extent>,
    ordered: Vec<Extent>,
    payload_len: u64,
    alignment: u64,
}

impl Layout {
    /// Build the layout from an already validated manifest.
    ///
    /// # Errors
    ///
    /// Rejects duplicate ids, extents outside the payload, zero lengths and
    /// overlapping extents. The manifest validator already enforces these; the
    /// layout re-checks because it is a separate consumer of the same data.
    pub fn from_manifest(manifest: &Manifest) -> Result<Self> {
        let mut by_shard = BTreeMap::new();
        let mut ordered = Vec::with_capacity(manifest.shards.len());
        for shard in &manifest.shards {
            if shard.length == 0 {
                return Err(StoreError::config("shard extent length is zero"));
            }
            let extent = Extent {
                shard: shard.shard_id,
                offset: shard.offset,
                len: shard.length,
            };
            if extent.end() > manifest.payload_len {
                return Err(StoreError::config(format!(
                    "shard {:?} ends at {} beyond payload {}",
                    extent.shard,
                    extent.end(),
                    manifest.payload_len
                )));
            }
            if by_shard.insert(extent.shard, extent).is_some() {
                return Err(StoreError::config("duplicate shard id in layout"));
            }
            ordered.push(extent);
        }
        ordered.sort_by_key(|extent| (extent.offset, extent.shard));
        for window in ordered.windows(2) {
            let [left, right] = window else {
                continue;
            };
            if left.end() > right.offset {
                return Err(StoreError::config("overlapping shard extents"));
            }
        }
        Ok(Self {
            by_shard,
            ordered,
            payload_len: manifest.payload_len,
            alignment: manifest.alignment,
        })
    }

    /// Payload length.
    #[must_use]
    pub const fn payload_len(&self) -> u64 {
        self.payload_len
    }

    /// Declared alignment.
    #[must_use]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Extent for a shard.
    #[must_use]
    pub fn extent(&self, id: &ShardId) -> Option<&Extent> {
        self.by_shard.get(id)
    }

    /// Extents in ascending physical order.
    #[must_use]
    pub fn ordered(&self) -> &[Extent] {
        &self.ordered
    }

    /// Largest extent, which bounds the minimum caller buffer.
    #[must_use]
    pub fn max_extent_bytes(&self) -> u64 {
        self.ordered
            .iter()
            .map(|extent| extent.len)
            .max()
            .unwrap_or(0)
    }

    /// Shard ids in ascending physical order, restricted to `selected` when a
    /// set is supplied.
    #[must_use]
    pub fn physical_order(
        &self,
        selected: Option<&std::collections::BTreeSet<ShardId>>,
    ) -> Vec<ShardId> {
        self.ordered
            .iter()
            .filter(|extent| selected.is_none_or(|set| set.contains(&extent.shard)))
            .map(|extent| extent.shard)
            .collect()
    }

    /// Coalesce adjacent extents into runs of at most `max_run_bytes`.
    ///
    /// Two extents are adjacent when the next one starts at or before
    /// `current_end + gap_bytes`. A run never splits a shard; when the next
    /// extent would exceed `max_run_bytes` it starts a new run. The result is
    /// deterministic for a given manifest and configuration.
    ///
    /// # Errors
    ///
    /// Rejects a zero `max_run_bytes`.
    pub fn runs(
        &self,
        selected: Option<&std::collections::BTreeSet<ShardId>>,
        max_run_bytes: u64,
        gap_bytes: u64,
    ) -> Result<Vec<Run>> {
        if max_run_bytes == 0 {
            return Err(StoreError::config("max_run_bytes is zero"));
        }
        let mut runs: Vec<Run> = Vec::new();
        for extent in &self.ordered {
            if let Some(set) = selected
                && !set.contains(&extent.shard)
            {
                continue;
            }
            match runs.last_mut() {
                Some(run) => {
                    let run_end = run.offset.saturating_add(run.length);
                    let projected_end = extent.offset.saturating_add(extent.len);
                    let adjacent = extent.offset <= run_end.saturating_add(gap_bytes);
                    let within_cap = projected_end.saturating_sub(run.offset) <= max_run_bytes;
                    if adjacent && within_cap {
                        let part_offset = extent.offset.saturating_sub(run.offset);
                        run.parts.push(RunPart {
                            shard: extent.shard,
                            offset: part_offset,
                            len: extent.len,
                        });
                        run.length = run.length.max(part_offset.saturating_add(extent.len));
                    } else {
                        runs.push(Run {
                            offset: extent.offset,
                            length: extent.len,
                            parts: vec![RunPart {
                                shard: extent.shard,
                                offset: 0,
                                len: extent.len,
                            }],
                        });
                    }
                }
                None => runs.push(Run {
                    offset: extent.offset,
                    length: extent.len,
                    parts: vec![RunPart {
                        shard: extent.shard,
                        offset: 0,
                        len: extent.len,
                    }],
                }),
            }
        }
        Ok(runs)
    }

    /// Summary used in receipts.
    #[must_use]
    pub fn report(&self) -> LayoutReport {
        let sum_extent_bytes: u64 = self.ordered.iter().map(|extent| extent.len).sum();
        let aligned = self
            .ordered
            .iter()
            .filter(|extent| {
                extent.offset % self.alignment == 0 && extent.len % self.alignment == 0
            })
            .count();
        let contiguous = self
            .ordered
            .windows(2)
            .filter(|window| window[0].end() == window[1].offset)
            .count();
        let runs = self
            .runs(None, u64::MAX, 0)
            .map(|runs| runs.len())
            .unwrap_or(0);
        LayoutReport {
            payload_len: self.payload_len,
            alignment: self.alignment,
            shard_count: self.ordered.len() as u64,
            min_extent_bytes: self
                .ordered
                .iter()
                .map(|extent| extent.len)
                .min()
                .unwrap_or(0),
            max_extent_bytes: self.max_extent_bytes(),
            sum_extent_bytes,
            payload_covered_bytes: sum_extent_bytes,
            padding_bytes: self.payload_len.saturating_sub(sum_extent_bytes),
            aligned_extents: aligned as u64,
            contiguous_pairs: contiguous as u64,
            coalesced_runs_unbounded: runs as u64,
        }
    }
}

/// Compact layout summary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutReport {
    /// Payload length.
    pub payload_len: u64,
    /// Declared alignment.
    pub alignment: u64,
    /// Extents.
    pub shard_count: u64,
    /// Smallest extent.
    pub min_extent_bytes: u64,
    /// Largest extent.
    pub max_extent_bytes: u64,
    /// Sum of extent lengths.
    pub sum_extent_bytes: u64,
    /// Bytes of the payload that belong to a shard.
    pub payload_covered_bytes: u64,
    /// Payload bytes that belong to no shard (unhashed padding).
    pub padding_bytes: u64,
    /// Extents whose offset and length are aligned.
    pub aligned_extents: u64,
    /// Adjacent extent pairs with no gap.
    pub contiguous_pairs: u64,
    /// Runs when every adjacent extent is merged.
    pub coalesced_runs_unbounded: u64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use selvedge_core::Sha256Digest;

    fn id(fill: u8) -> ShardId {
        ShardId::from_digest(Sha256Digest::of(&[fill]))
    }

    fn extent(fill: u8, offset: u64, len: u64) -> Extent {
        Extent {
            shard: id(fill),
            offset,
            len,
        }
    }

    fn layout(extents: Vec<Extent>, payload_len: u64, alignment: u64) -> Layout {
        let by_shard = extents
            .iter()
            .map(|extent| (extent.shard, *extent))
            .collect::<BTreeMap<_, _>>();
        let mut ordered = extents;
        ordered.sort_by_key(|extent| (extent.offset, extent.shard));
        Layout {
            by_shard,
            ordered,
            payload_len,
            alignment,
        }
    }

    #[test]
    fn adjacent_extents_coalesce_and_long_runs_split_by_shard() {
        let layout = layout(
            vec![
                extent(1, 4096, 4096),
                extent(2, 8192, 4096),
                extent(3, 12288, 4096),
                extent(4, 65536, 4096),
            ],
            69632,
            4096,
        );
        let runs = layout.runs(None, 12288, 0).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].parts.len(), 3);
        assert_eq!(runs[0].length, 12288);
        assert_eq!(runs[1].parts.len(), 1);
    }

    #[test]
    fn a_gap_start_new_run_unless_permitted() {
        let layout = layout(vec![extent(1, 0, 4096), extent(2, 8192, 4096)], 12288, 4096);
        assert_eq!(layout.runs(None, 1 << 20, 0).unwrap().len(), 2);
        assert_eq!(layout.runs(None, 1 << 20, 4096).unwrap().len(), 1);
    }

    #[test]
    fn physical_order_is_independent_of_manifest_order() {
        let layout = layout(vec![extent(2, 4096, 4096), extent(1, 0, 4096)], 8192, 4096);
        let order = layout.physical_order(None);
        assert_eq!(order, vec![id(1), id(2)]);
    }

    #[test]
    fn bounded_runs_never_split_a_shard() {
        let layout = layout(
            vec![
                extent(1, 0, 4096),
                extent(2, 4096, 4096),
                extent(3, 8192, 4096),
            ],
            12288,
            4096,
        );
        let runs = layout.runs(None, 8192, 0).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].parts.len(), 2);
        assert_eq!(runs[1].parts.len(), 1);
    }
}
