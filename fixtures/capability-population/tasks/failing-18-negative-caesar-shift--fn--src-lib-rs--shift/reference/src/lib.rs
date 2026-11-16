//! Caesar rotation.

pub fn shift(text: &str, amount: i32) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_lowercase() {
                let offset = (character as i32 - 'a' as i32 + amount)
                    .rem_euclid(26) as u8;
                (b'a' + offset) as char
            } else if character.is_ascii_uppercase() {
                let offset = (character as i32 - 'A' as i32 + amount)
                    .rem_euclid(26) as u8;
                (b'A' + offset) as char
            } else {
                character
            }
        })
        .collect()
}

#[cfg(test)]
mod visible {
    use crate::shift;

    #[test]
    fn shifts_backwards_across_the_alphabet() {
        assert_eq!(shift("abc", -1), "zab");
    }

    #[test]
    fn preserves_case_and_punctuation() {
        assert_eq!(shift("Hello, World!", 3), "Khoor, Zruog!");
    }

    #[test]
    fn wraps_forwards_and_backwards() {
        assert_eq!(shift("xyz", 3), "abc");
        assert_eq!(shift("ABC", -3), "XYZ");
    }

    #[test]
    fn a_zero_shift_changes_nothing() {
        assert_eq!(shift("a1!", 0), "a1!");
    }
}
