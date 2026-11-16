//! Largest gap between consecutive values.

pub fn max_gap(values: &[i64]) -> i64 {
    let mut best = values[1] - values[0];
    for pair in values.windows(2) {
        let gap = pair[1] - pair[0];
        if gap > best {
            best = gap;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::max_gap;

    #[test]
    fn single_value_has_no_gap() {
        assert_eq!(max_gap(&[5]), 0);
    }
}
