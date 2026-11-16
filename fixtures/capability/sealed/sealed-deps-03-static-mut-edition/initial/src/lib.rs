//! A mutable static counter. This does not build under edition 2024.

static mut COUNTER: i64 = 0;

pub fn bump() -> i64 {
    let reference = unsafe { &COUNTER };
    unsafe { COUNTER += 1; }
    *reference
}
