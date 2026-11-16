#[cfg(test)]
mod tests {
    #[test]
    fn allocates_past_the_bound() {
        let mut blocks: Vec<Vec<u8>> = Vec::new();
        for _ in 0..256 {
            blocks.push(vec![0u8; 8 * 1024 * 1024]);
        }
        assert_eq!(blocks.len(), 256);
    }
}
