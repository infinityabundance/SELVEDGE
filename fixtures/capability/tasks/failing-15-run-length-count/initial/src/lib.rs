//! Run-length encoding. The visible test fails: the counter restarts
//! at zero after a run is emitted.

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
            count = 0;
        }
    }
    out.push(current);
    out.push_str(&count.to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::encode;

    #[test]
    fn visible_two_as() {
        assert_eq!(encode("aab"), "a2b1");
    }
}
