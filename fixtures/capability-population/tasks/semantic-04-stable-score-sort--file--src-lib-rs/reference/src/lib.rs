//! Score ranking with a deterministic tie-break.

pub fn ranked(entries: &[(String, u32)]) -> Vec<String> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| {
        right.1.cmp(&left.1).then(left.0.cmp(&right.0))
    });
    sorted.into_iter().map(|(name, _score)| name).collect()
}
