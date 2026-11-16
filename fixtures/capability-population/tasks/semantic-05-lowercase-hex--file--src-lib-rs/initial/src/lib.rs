//! Hex encoding with an upper-case format specifier.

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02X}")).collect()
}

#[cfg(test)]
mod tests {
    use super::hex;

    #[test]
    fn digits_only_are_the_same_in_either_case() {
        assert_eq!(hex(&[1, 2]), "0102");
    }
}
