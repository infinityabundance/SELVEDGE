//! A counter closure.

pub fn make_counter(seed: u64) -> impl FnMut() -> u64 {
    let mut value = seed;
    move || {
        value += 1;
        value
    }
}
