//! Score ranking without a tie-break.

pub fn ranked(entries: &[(String, u32)]) -> Vec<String> {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| right.1.cmp(&left.1));
    sorted.into_iter().map(|(name, _score)| name).collect()
}

#[cfg(test)]
mod tests {
    use super::ranked;

    #[test]
    fn orders_distinct_scores() {
        let entries = vec![("low".to_string(), 1), ("high".to_string(), 2)];
        assert_eq!(ranked(&entries), vec!["high", "low"]);
    }
}
