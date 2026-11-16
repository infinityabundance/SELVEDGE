//! Push a value and report the value that was last before it.

pub fn push_and_report(values: &mut Vec<i32>, value: i32) -> i32 {
    let last = values.last().copied().unwrap_or(0);
    values.push(value);
    last
}
