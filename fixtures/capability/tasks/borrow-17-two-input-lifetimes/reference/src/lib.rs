//! Choose the longer of two texts.

pub fn longer<'a>(left: &'a str, right: &'a str) -> &'a str {
    if right.len() > left.len() {
        right
    } else {
        left
    }
}
