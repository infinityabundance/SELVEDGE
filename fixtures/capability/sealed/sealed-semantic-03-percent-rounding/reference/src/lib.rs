//! Whole-number percentages, rounded with ties upward.

pub fn percent(part: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    (part * 100 + total / 2) / total
}
