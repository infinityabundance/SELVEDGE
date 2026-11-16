//! Increment every entry in place.

pub fn stamp(entries: &mut Vec<u32>) -> u32 {
    
    for index in 0..entries.len() {
        entries[index] += 1;
    }
    entries.iter().sum()

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn increments_every_entry_and_sums() {
        let mut entries = vec![1, 2];
        assert_eq!(stamp(&mut entries), 5);
        assert_eq!(entries, vec![2, 3]);
    }
}
