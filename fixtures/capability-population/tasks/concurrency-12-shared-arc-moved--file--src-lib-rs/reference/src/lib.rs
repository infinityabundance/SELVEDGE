//! A chunked sum.

use std::sync::Arc;

pub fn chunked_sum(values: Arc<Vec<u64>>, threads: usize) -> u64 {
    let mut handles = Vec::new();
    for index in 0..threads {
        let values = Arc::clone(&values);
        handles.push(std::thread::spawn(move || {
            values.iter().skip(index).step_by(threads).sum::<u64>()
        }));
    }
    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
}
