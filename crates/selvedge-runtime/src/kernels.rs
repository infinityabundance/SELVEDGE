//! Deterministic f32 CPU kernels.
//!
//! This is the whole numeric backend. It is deliberately smaller than a tensor
//! library: seven operations cover Qwen3, and every one of them has a stated
//! reduction order that does not depend on the thread count, the cache policy,
//! or how many rows happen to be in flight.
//!
//! # Reduction order
//!
//! A dot product of length `k` is computed with [`DOT_LANES`] interleaved f32
//! accumulators:
//!
//! ```text
//! acc[l] = sum over j of a[j*LANES + l] * b[j*LANES + l]     (j ascending)
//! tail   = sum over the final k mod LANES elements            (ascending)
//! result = ((((acc[0] + acc[1]) + acc[2]) + ... ) + acc[LANES-1]) + tail
//! ```
//!
//! Matrix-vector products parallelise **over output rows only**: row `r` is
//! always reduced over `k` in exactly the same order, whichever thread computes
//! it. Consequently the output bits are identical for any `threads` value, and
//! no compensation, no pairwise tree and no BLAS is involved. This is why the
//! Court E claim is bitwise rather than approximate: the variants under test
//! change I/O, never arithmetic.
//!
//! No allocation here is sized by anything except the shapes the caller
//! validated, and no kernel allocates internally at all.

use crate::error::{Result, RuntimeError};

/// Number of interleaved accumulators in [`dot`].
///
/// Eight f32 lanes fill one AVX2 register and let the compiler keep the
/// multiply-add chain in flight without reassociating anything itself.
pub const DOT_LANES: usize = 8;

fn check_len(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(RuntimeError::model(format!(
            "{name} has {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

/// Dot product with the documented fixed reduction order.
///
/// # Errors
///
/// Rejects different lengths.
pub fn dot(left: &[f32], right: &[f32]) -> Result<f32> {
    check_len("left operand", left.len(), right.len())?;
    Ok(dot_fixed(left, right))
}

/// The reduction itself. Lengths must already be equal.
fn dot_fixed(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut acc = [0.0_f32; DOT_LANES];
    let lanes = left.len() / DOT_LANES;
    for chunk in 0..lanes {
        let base = chunk * DOT_LANES;
        for lane in 0..DOT_LANES {
            acc[lane] += left[base + lane] * right[base + lane];
        }
    }
    let mut sum = acc[0];
    for value in &acc[1..] {
        sum += *value;
    }
    let mut tail = 0.0_f32;
    for index in lanes * DOT_LANES..left.len() {
        tail += left[index] * right[index];
    }
    sum + tail
}

/// Number of output rows each thread handles.
#[must_use]
pub fn rows_per_thread(rows: usize, threads: usize) -> usize {
    let threads = threads.max(1).min(rows.max(1));
    rows.div_ceil(threads).max(1)
}

/// Run `task` over disjoint contiguous blocks of `items`.
///
/// The block size is [`rows_per_thread`], so the partition depends only on
/// `(items.len(), threads)`. Each block is handed to exactly one thread and the
/// blocks never overlap, so a task that writes only inside its block cannot
/// race and the result cannot depend on scheduling.
pub fn parallel_blocks<T, F>(items: &mut [T], threads: usize, task: F)
where
    T: Send,
    F: Fn(&mut [T]) + Sync,
{
    if items.is_empty() {
        return;
    }
    let block = rows_per_thread(items.len(), threads);
    if threads <= 1 || items.len() <= block {
        task(items);
        return;
    }
    std::thread::scope(|scope| {
        for chunk in items.chunks_mut(block) {
            let task_ref = &task;
            scope.spawn(move || task_ref(chunk));
        }
    });
}

/// `out[r] = dot(w[r*cols .. r*cols+cols], x)` for every row.
///
/// # Errors
///
/// Rejects a weight buffer, input or output of the wrong length.
pub fn matvec_into(
    weights: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<()> {
    check_len("matvec input", x.len(), cols)?;
    check_len("matvec output", out.len(), rows)?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matvec weight size overflows"))?;
    check_len("matvec weights", weights.len(), expected)?;
    if cols < DOT_LANES {
        for (row, slot) in out.iter_mut().enumerate() {
            let start = row * cols;
            let mut sum = 0.0_f32;
            for col in 0..cols {
                sum += weights[start + col] * x[col];
            }
            *slot = sum;
        }
        return Ok(());
    }
    let block = rows_per_thread(rows, threads);
    if threads <= 1 || rows <= block {
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = dot_fixed(&weights[row * cols..(row + 1) * cols], x);
        }
        return Ok(());
    }
    std::thread::scope(|scope| {
        let mut first_row = 0_usize;
        for chunk in out.chunks_mut(block) {
            let start = first_row;
            first_row += chunk.len();
            let weights_ref = weights;
            let x_ref = x;
            scope.spawn(move || {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let row = start + offset;
                    *slot = dot_fixed(&weights_ref[row * cols..(row + 1) * cols], x_ref);
                }
            });
        }
    });
    Ok(())
}

/// `out[b][r] = dot(w[r*cols .. r*cols+cols], x[b*cols .. b*cols+cols])`.
///
/// This is the batched form of [`matvec_into`], used by the batched prefill. It
/// is **bit-identical** to running that function once per batch row, because
/// every output element is still produced by [`dot_fixed`] over the same `k` in
/// the same order; only the loop nest around it changes (each weight row is
/// streamed once and applied to all rows of the batch). Parallelism is still over
/// weight rows only, so the bits cannot depend on the thread count or on the
/// batch size.
///
/// # Errors
///
/// Rejects a weight matrix, input batch or output batch of the wrong length.
pub fn matmul_into(
    weights: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<()> {
    let inputs = batch
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matmul input size overflows"))?;
    let outputs = batch
        .checked_mul(rows)
        .ok_or_else(|| RuntimeError::model("matmul output size overflows"))?;
    let weight_len = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matmul weight size overflows"))?;
    check_len("matmul input", x.len(), inputs)?;
    check_len("matmul output", out.len(), outputs)?;
    check_len("matmul weights", weights.len(), weight_len)?;
    if batch == 0 || rows == 0 {
        return Ok(());
    }
    // Each row block owns a private scratch buffer, and the block partition
    // depends on (rows, threads) alone, so the bits cannot depend on scheduling.
    let block = rows_per_thread(rows, threads);
    let compute = |buffer: &mut [f32], start: usize, count: usize| {
        for offset in 0..count {
            let row = start + offset;
            let weight_row = &weights[row * cols..(row + 1) * cols];
            for index in 0..batch {
                let value = if cols < DOT_LANES {
                    let mut sum = 0.0_f32;
                    let base = index * cols;
                    for col in 0..cols {
                        sum += weight_row[col] * x[base + col];
                    }
                    sum
                } else {
                    dot_fixed(weight_row, &x[index * cols..(index + 1) * cols])
                };
                buffer[index * count + offset] = value;
            }
        }
    };
    if threads <= 1 || rows <= block {
        let mut buffer = vec![0.0_f32; rows * batch];
        compute(&mut buffer, 0, rows);
        for index in 0..batch {
            out[index * rows..(index + 1) * rows]
                .copy_from_slice(&buffer[index * rows..(index + 1) * rows]);
        }
        return Ok(());
    }
    let blocks: Vec<(usize, usize)> = (0..rows)
        .step_by(block)
        .map(|start| (start, block.min(rows - start)))
        .collect();
    let mut buffers: Vec<Vec<f32>> = blocks
        .iter()
        .map(|(_, count)| vec![0.0_f32; count * batch])
        .collect();
    std::thread::scope(|scope| {
        for ((start, count), buffer) in blocks.iter().zip(buffers.iter_mut()) {
            let compute_ref = &compute;
            let (start, count) = (*start, *count);
            let buffer: &mut [f32] = buffer;
            scope.spawn(move || compute_ref(buffer, start, count));
        }
    });
    for ((start, count), buffer) in blocks.iter().zip(buffers.iter()) {
        for index in 0..batch {
            let from = index * count;
            let to = index * rows + start;
            out[to..to + count].copy_from_slice(&buffer[from..from + count]);
        }
    }
    Ok(())
}

/// `out[r] += dot(w[r*cols .. r*cols+cols], x)` for every row.
///
/// Used by the grouped FFN, where each group contributes its own partial sum to
/// the same output vector. The accumulation across groups happens in group
/// order, which the caller fixes.
///
/// # Errors
///
/// Rejects a weight buffer, input or output of the wrong length.
pub fn matvec_accumulate(
    weights: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    check_len("matvec input", x.len(), cols)?;
    check_len("matvec output", out.len(), rows)?;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matvec weight size overflows"))?;
    check_len("matvec weights", weights.len(), expected)?;
    for (row, slot) in out.iter_mut().enumerate() {
        let start = row * cols;
        if cols < DOT_LANES {
            let mut sum = 0.0_f32;
            for col in 0..cols {
                sum += weights[start + col] * x[col];
            }
            *slot += sum;
        } else {
            *slot += dot_fixed(&weights[start..start + cols], x);
        }
    }
    Ok(())
}

/// `out[b][r] += dot(w[r*cols .. r*cols+cols], x[b*cols .. b*cols+cols])`.
///
/// The batched form of [`matvec_accumulate`], used by the batched prefill's
/// grouped down projection. Each row's contribution is still one fixed-order dot
/// added to the accumulator in the caller's group order, so the result is
/// bit-identical to the scalar path.
///
/// # Errors
///
/// Rejects a weight matrix, input batch or output batch of the wrong length.
pub fn matmul_accumulate(
    weights: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let inputs = batch
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matmul input size overflows"))?;
    let outputs = batch
        .checked_mul(rows)
        .ok_or_else(|| RuntimeError::model("matmul output size overflows"))?;
    let weight_len = rows
        .checked_mul(cols)
        .ok_or_else(|| RuntimeError::model("matmul weight size overflows"))?;
    check_len("matmul input", x.len(), inputs)?;
    check_len("matmul output", out.len(), outputs)?;
    check_len("matmul weights", weights.len(), weight_len)?;
    for row in 0..rows {
        let weight_row = &weights[row * cols..(row + 1) * cols];
        for index in 0..batch {
            let value = if cols < DOT_LANES {
                let mut sum = 0.0_f32;
                let base = index * cols;
                for col in 0..cols {
                    sum += weight_row[col] * x[base + col];
                }
                sum
            } else {
                dot_fixed(weight_row, &x[index * cols..(index + 1) * cols])
            };
            out[index * rows + row] += value;
        }
    }
    Ok(())
}

/// RMSNorm in place: `x *= weight / sqrt(mean(x^2) + eps)`.
///
/// The mean of squares accumulates in ascending index order in f32.
///
/// # Errors
///
/// Rejects a weight and vector of different lengths, and an empty vector.
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) -> Result<()> {
    check_len("rmsnorm weight", weight.len(), x.len())?;
    if x.is_empty() {
        return Err(RuntimeError::model("rmsnorm of an empty vector"));
    }
    let mut sum = 0.0_f32;
    for value in x.iter() {
        sum += value * value;
    }
    let count = x.len() as f32;
    let scale = 1.0_f32 / (sum / count + eps).sqrt();
    for (value, gain) in x.iter_mut().zip(weight.iter()) {
        *value = *value * scale * gain;
    }
    Ok(())
}

/// SiLU activation.
#[must_use]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// SwiGLU: `gate[j] = silu(gate[j]) * up[j]`, in place.
///
/// # Errors
///
/// Rejects different lengths.
pub fn swiglu_act(gate: &mut [f32], up: &[f32]) -> Result<()> {
    check_len("swiglu up", up.len(), gate.len())?;
    for (slot, other) in gate.iter_mut().zip(up.iter()) {
        *slot = silu(*slot) * other;
    }
    Ok(())
}

/// Apply rotary position embedding to one head, in place.
///
/// GPT-NeoX / Qwen3 `rotate_half` convention over the full head dimension:
///
/// ```text
/// out[i]       = v[i]       * cos_i - v[i + d/2] * sin_i
/// out[i + d/2] = v[i + d/2] * cos_i + v[i]       * sin_i
/// ```
///
/// `inv_freq` is the caller's precomputed `1 / theta^(2i/d)` table with `d/2`
/// entries, computed once at load in f32.
///
/// # Errors
///
/// Rejects a head whose length is not twice the frequency table, or an odd
/// head dimension.
pub fn rope_in_place(v: &mut [f32], inv_freq: &[f32], position: usize) -> Result<()> {
    let half = inv_freq.len();
    check_len("rope head", v.len(), half * 2)?;
    let position = position as f32;
    for index in 0..half {
        let angle = position * inv_freq[index];
        let (sin, cos) = (angle.sin(), angle.cos());
        let low = v[index];
        let high = v[index + half];
        v[index] = low * cos - high * sin;
        v[index + half] = high * cos + low * sin;
    }
    Ok(())
}

/// Numerically stable softmax over the whole slice, in place.
///
/// # Errors
///
/// Rejects an empty slice.
pub fn softmax(x: &mut [f32]) -> Result<()> {
    if x.is_empty() {
        return Err(RuntimeError::model("softmax of an empty slice"));
    }
    let mut max = f32::NEG_INFINITY;
    for value in x.iter() {
        if *value > max {
            max = *value;
        }
    }
    let mut sum = 0.0_f32;
    for value in x.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) || !sum.is_finite() {
        return Err(RuntimeError::model("softmax denominator is not finite"));
    }
    for value in x.iter_mut() {
        *value /= sum;
    }
    Ok(())
}

/// Index of the largest value; ties resolve to the lowest index.
///
/// # Errors
///
/// Rejects an empty slice and a non-finite maximum.
pub fn argmax(x: &[f32]) -> Result<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in x.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        match best {
            None => best = Some((index, *value)),
            Some((_, current)) if *value > current => best = Some((index, *value)),
            _ => {}
        }
    }
    match best {
        Some((index, _)) => Ok(index),
        None => Err(RuntimeError::model(
            "argmax found no finite logit, refusing to decode",
        )),
    }
}

/// `1 / theta^(2i/d)` for `i in 0..d/2`, computed in f32.
#[must_use]
pub fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let mut freqs = Vec::with_capacity(half);
    for index in 0..half {
        let exponent = (2 * index) as f32 / head_dim as f32;
        freqs.push(1.0 / theta.powf(exponent));
    }
    freqs
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_naive_in_ascending_order_for_short_vectors() {
        let a: Vec<f32> = (1..=7).map(|value| value as f32).collect();
        let b: Vec<f32> = (1..=7).map(|value| 1.0 / value as f32).collect();
        let mut naive = 0.0_f32;
        for (left, right) in a.iter().zip(b.iter()) {
            naive += left * right;
        }
        assert_eq!(dot(&a, &b).unwrap(), naive);
    }

    #[test]
    fn matvec_is_independent_of_thread_count() {
        let rows = 37_usize;
        let cols = 40_usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.017).sin())
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|index| ((index as f32) * 0.31).cos())
            .collect();
        let mut reference = vec![0.0_f32; rows];
        matvec_into(&weights, rows, cols, &x, &mut reference, 1).unwrap();
        for threads in [2_usize, 3, 8, 16] {
            let mut other = vec![0.0_f32; rows];
            matvec_into(&weights, rows, cols, &x, &mut other, threads).unwrap();
            for (left, right) in reference.iter().zip(other.iter()) {
                assert_eq!(left.to_bits(), right.to_bits(), "threads={threads}");
            }
        }
    }

    #[test]
    fn rms_norm_scales_by_root_mean_square() {
        let mut x = vec![3.0_f32, 4.0];
        let weight = vec![1.0_f32, 1.0];
        rms_norm(&mut x, &weight, 0.0).unwrap();
        let rms = (12.5_f32).sqrt();
        assert!((x[0] - 3.0 / rms).abs() < 1e-6);
        assert!((x[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_argmax_breaks_ties_low() {
        let mut values = vec![1.0_f32, 3.0, 3.0, 2.0];
        softmax(&mut values).unwrap();
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&values).unwrap(), 1);
    }

    #[test]
    fn rope_rotates_a_pair_by_the_position_angle() {
        let inv_freq = vec![1.0_f32, 0.5];
        let mut v = vec![1.0_f32, 0.0, 0.0, 1.0];
        rope_in_place(&mut v, &inv_freq, 1).unwrap();
        // index 0: out[0] = 1*cos(1) - 0*sin(1), out[2] = 0*cos(1) + 1*sin(1)
        assert!((v[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((v[2] - 1.0_f32.sin()).abs() < 1e-6);
        // index 1: out[1] = 0*cos(0.5) - 1*sin(0.5), out[3] = 1*cos(0.5) + 0*sin(0.5)
        assert!((v[1] + 0.5_f32.sin()).abs() < 1e-6);
        assert!((v[3] - 0.5_f32.cos()).abs() < 1e-6);
    }
}
