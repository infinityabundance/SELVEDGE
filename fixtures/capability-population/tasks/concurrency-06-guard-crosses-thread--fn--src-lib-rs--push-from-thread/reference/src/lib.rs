//! Push from a spawned thread.

use std::sync::{Arc, Mutex};
use std::thread;

pub fn push_from_thread(values: &Arc<Mutex<Vec<i32>>>, value: i32) {
    let values = Arc::clone(values);
    thread::spawn(move || {
        values.lock().unwrap().push(value);
    })
    .join()
    .unwrap();
}

#[cfg(test)]
mod visible {
    use crate::push_from_thread;
    use std::sync::{Arc, Mutex};

    #[test]
    fn pushes_one_value() {
        let values = Arc::new(Mutex::new(Vec::new()));
        push_from_thread(&values, 5);
        assert_eq!(*values.lock().unwrap(), vec![5]);
    }

    #[test]
    fn pushes_successive_values() {
        let values = Arc::new(Mutex::new(Vec::new()));
        push_from_thread(&values, 1);
        push_from_thread(&values, 2);
        assert_eq!(*values.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn separate_arcs_are_independent() {
        let first = Arc::new(Mutex::new(Vec::new()));
        let second = Arc::new(Mutex::new(Vec::<i32>::new()));
        push_from_thread(&first, 7);
        assert_eq!(*first.lock().unwrap(), vec![7]);
        assert!(second.lock().unwrap().is_empty());
    }
}
