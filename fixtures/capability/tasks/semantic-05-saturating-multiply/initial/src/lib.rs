//! Scaling, currently unchecked.

pub fn scale(value: u32, factor: u32) -> u32 {
    value * factor
}

#[cfg(test)]
mod tests {
    use super::scale;

    #[test]
    fn small_products_are_exact() {
        assert_eq!(scale(2, 3), 6);
    }
}
