//! Run-length encoding.

pub fn encode(text: &str) -> String {
    let mut out = String::new();
    let mut characters = text.chars();
    let mut current = match characters.next() {
        Some(character) => character,
        None => return out,
    };
    let mut count = 1u32;
    for character in characters {
        if character == current {
            count += 1;
        } else {
            out.push(current);
            out.push_str(&count.to_string());
            current = character;
            count = 1;
        }
    }
    out.push(current);
    out.push_str(&count.to_string());
    out
}

#[cfg(test)]
mod visible {
    use crate::encode;

    #[test]
    fn every_run_carries_its_length() {
        assert_eq!(encode("aab"), "a2b1");
        assert_eq!(encode("abbcccd"), "a1b2c3d1");
    }

    #[test]
    fn a_single_character_is_a_run_of_one() {
        assert_eq!(encode("a"), "a1");
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(encode(""), "");
    }

    #[test]
    fn a_long_run_is_counted_exactly() {
        assert_eq!(encode("aaaa"), "a4");
    }
}
