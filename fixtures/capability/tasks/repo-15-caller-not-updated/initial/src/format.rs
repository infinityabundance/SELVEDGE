//! Ratio formatting.

pub fn ratio(part: u32, total: u32) -> String {
    if total == 0 {
        return "0.0%".to_string();
    }
    format!("{:.1}%", part as f64 * 100.0 / total as f64)
}
