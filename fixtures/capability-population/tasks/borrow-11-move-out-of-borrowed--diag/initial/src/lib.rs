//! A sorted copy of a borrowed list. The crate does not build: it
//! moves the vector out of a shared reference.

pub fn sorted_copy(values: &Vec<String>) -> Vec<String> {
    let mut owned = *values;
    owned.sort();
    owned
}
