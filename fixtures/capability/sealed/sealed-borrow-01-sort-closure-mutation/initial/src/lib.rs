//! Duplicate counting before a sort. The crate does not build: the
//! comparator mutates the vector being sorted.

pub fn sort_and_count_duplicates(values: &mut Vec<i64>) -> usize {
    values.sort_by(|left, right| {
        values.push(*left);
        left.cmp(right)
    });
    values.windows(2).filter(|pair| pair[0] == pair[1]).count()
}
