#![deny(unsafe_op_in_unsafe_fn)]

//! Raw-pointer access under edition 2024.

/// # Safety
///
/// `pointer` must be valid for one read.
pub unsafe fn read_byte(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}
