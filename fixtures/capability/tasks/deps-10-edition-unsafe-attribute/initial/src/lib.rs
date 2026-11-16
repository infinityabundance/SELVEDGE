//! An exported entry point. This does not build under edition 2024.

#[no_mangle]
pub extern "C" fn answer() -> i32 {
    42
}
