//! Choose the longer of two texts. The crate does not build: the
//! result may borrow from the second input.

pub fn longer<'a, 'b>(left: &'a str, right: &'b str) -> &'a str {
    if right.len() > left.len() {
        right
    } else {
        left
    }
}
