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
