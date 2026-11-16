//! Percentage clamping.

pub fn clamp_percent(part: u32, total: u32) -> u32 {
    if total == 0 {
        return 0;
    }
    part.min(total) * 100 / total
}
