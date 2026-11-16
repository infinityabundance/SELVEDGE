//! Lowercase hex encoding.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::hex;

    #[test]
    fn uses_lowercase_letters() {
        assert_eq!(hex(&[0xab, 0x0f]), "ab0f");
    }

    #[test]
    fn encodes_a_word() {
        assert_eq!(hex(&[0xde, 0xad]), "dead");
    }

    #[test]
    fn empty_is_empty() {
        assert_eq!(hex(&[]), "");
    }

    #[test]
    fn pads_every_byte_to_two_digits() {
        assert_eq!(hex(&[0, 1, 255]), "0001ff");
    }
}
