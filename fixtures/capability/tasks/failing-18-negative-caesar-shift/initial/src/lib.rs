//! Caesar rotation. The visible test fails: a negative amount is
//! converted to `u8` before the rotation.

pub fn shift(text: &str, amount: i32) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_lowercase() {
                let offset = (character as u8 - b'a').wrapping_add(amount as u8) % 26;
                (b'a' + offset) as char
            } else if character.is_ascii_uppercase() {
                let offset = (character as u8 - b'A').wrapping_add(amount as u8) % 26;
                (b'A' + offset) as char
            } else {
                character
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::shift;

    #[test]
    fn visible_backwards() {
        assert_eq!(shift("abc", -1), "zab");
    }
}
