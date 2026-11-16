//! Greedy word wrapping.

pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() > width {
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
mod visible {
    use crate::wrap;

    #[test]
    fn a_space_counts_against_the_width() {
        assert_eq!(wrap("aaa bbb", 6), vec!["aaa", "bbb"]);
    }

    #[test]
    fn several_words_fill_one_line() {
        assert_eq!(wrap("aa bb cc", 5), vec!["aa bb", "cc"]);
    }

    #[test]
    fn empty_input_has_no_lines() {
        assert!(wrap("", 4).is_empty());
    }

    #[test]
    fn an_overlong_word_stays_whole() {
        assert_eq!(wrap("abcdef", 3), vec!["abcdef"]);
    }
}
