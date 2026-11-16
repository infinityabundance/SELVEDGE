//! Render bytes as decimal text through a local newtype.

pub struct Bytes(pub Vec<u8>);

impl core::fmt::Display for Bytes {
// TODO(FUNCTION)
}

pub fn render(values: &[u8]) -> String {
    Bytes(values.to_vec()).to_string()
}

#[cfg(test)]
mod visible {
    use crate::render;

    #[test]
    fn joins_decimal_bytes() {
        assert_eq!(render(&[1, 2, 3]), "1,2,3");
    }

    #[test]
    fn empty_is_empty() {
        assert_eq!(render(&[]), "");
    }

    #[test]
    fn single_byte_has_no_separator() {
        assert_eq!(render(&[255]), "255");
    }

    #[test]
    fn zeros_are_rendered() {
        assert_eq!(render(&[0, 0]), "0,0");
    }
}
