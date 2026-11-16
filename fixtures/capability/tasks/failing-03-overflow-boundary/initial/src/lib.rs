//! Running total over bytes.

pub fn running_total(values: &[u8]) -> u8 {
    let mut total = 0u8;
    for value in values {
        total += *value;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::running_total;

    #[test]
    fn saturates_instead_of_overflowing() {
        assert_eq!(running_total(&[250, 10]), 255);
    }
}
