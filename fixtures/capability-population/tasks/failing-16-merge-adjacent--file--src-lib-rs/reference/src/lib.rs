//! Interval merging.

pub fn merge(ranges: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|range| range.0);
    let mut out: Vec<(i32, i32)> = Vec::new();
    for range in sorted {
        match out.last_mut() {
            Some(current) if range.0 <= current.1 => {
                if range.1 > current.1 {
                    current.1 = range.1;
                }
            }
            _ => out.push(range),
        }
    }
    out
}
