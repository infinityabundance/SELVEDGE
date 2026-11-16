//! The conversion trait shared by the modules.

pub trait AsI64 {
    fn as_i64(&self) -> i64;
}

impl AsI64 for i32 {
    fn as_i64(&self) -> i64 {
        i64::from(*self)
    }
}

impl AsI64 for i64 {
    fn as_i64(&self) -> i64 {
        *self
    }
}
