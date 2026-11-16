//! Duplicate counting before a sort.

pub fn sort_and_count_duplicates(values: &mut Vec<i64>) -> usize {
    values.sort();
    values.windows(2).filter(|pair| pair[0] == pair[1]).count()
}
