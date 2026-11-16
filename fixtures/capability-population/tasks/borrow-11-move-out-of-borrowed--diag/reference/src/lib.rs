//! A sorted copy of a borrowed list.

pub fn sorted_copy(values: &Vec<String>) -> Vec<String> {
    let mut owned = values.clone();
    owned.sort();
    owned
}
