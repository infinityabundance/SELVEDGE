//! Shortening with a trailing ellipsis.

pub fn shorten(text: &str, limit: usize) -> String {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() <= limit {
        return text.to_string();
    }
    let mut out: String = characters.iter().take(limit).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::shorten;

    #[test]
    fn visible_exact_fit() {
        assert_eq!(shorten("abc", 3), "abc");
    }
}
