//! Shortening with a trailing ellipsis inside the budget.

pub fn shorten(text: &str, limit: usize) -> String {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() <= limit {
        return text.to_string();
    }
    if limit == 0 {
        return String::new();
    }
    let mut out: String = characters.iter().take(limit - 1).collect();
    out.push('…');
    out
}
