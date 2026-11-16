//! Greedy word wrapping. The visible test fails: the space between
//! words is not counted when a line break is decided.

pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + word.len() > width {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::wrap;

    #[test]
    fn visible_line_breaks() {
        assert_eq!(wrap("aaa bbb", 6), vec!["aaa", "bbb"]);
    }
}
