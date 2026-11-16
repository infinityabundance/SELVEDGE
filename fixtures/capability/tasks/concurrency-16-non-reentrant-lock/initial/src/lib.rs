//! Read a mutex-protected value twice. The crate builds, but the
//! function takes the same non-reentrant lock twice.

use std::sync::Mutex;

pub fn read_twice(shared: &Mutex<u32>) -> u32 {
    let first = shared.lock().unwrap();
    let second = shared.lock().unwrap();
    *first + *second
}

#[cfg(test)]
mod tests {
    use super::read_twice;
    use std::sync::Mutex;

    #[test]
    fn visible_read() {
        let shared = Mutex::new(21);
        assert_eq!(read_twice(&shared), 42);
    }
}
