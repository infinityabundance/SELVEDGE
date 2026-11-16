//! Read a mutex-protected value twice.

use std::sync::Mutex;

pub fn read_twice(shared: &Mutex<u32>) -> u32 {
    let value = *shared.lock().unwrap();
    value * 2
}

#[cfg(test)]
mod visible {
    use std::sync::Mutex;

    use crate::read_twice;

    #[test]
    fn doubles_the_value() {
        let shared = Mutex::new(21);
        assert_eq!(read_twice(&shared), 42);
    }

    #[test]
    fn zero_doubles_to_zero() {
        let shared = Mutex::new(0);
        assert_eq!(read_twice(&shared), 0);
    }

    #[test]
    fn the_lock_is_released_afterwards() {
        let shared = Mutex::new(4);
        assert_eq!(read_twice(&shared), 8);
        assert_eq!(*shared.lock().unwrap(), 4);
    }
}
