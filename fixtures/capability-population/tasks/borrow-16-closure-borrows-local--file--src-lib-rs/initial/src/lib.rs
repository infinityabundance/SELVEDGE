//! A counter closure. The crate does not build: the returned closure
//! borrows a local binding.

pub fn make_counter(seed: u64) -> impl FnMut() -> u64 {
    let mut value = seed;
    || {
        value += 1;
        value
    }
}
