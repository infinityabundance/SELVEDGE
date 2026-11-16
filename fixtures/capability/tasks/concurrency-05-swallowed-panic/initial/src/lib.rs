//! Scoped workers whose panic is swallowed.

use std::thread;

pub fn run_workers(jobs: &[i32]) -> Result<i32, String> {
    let mut total = 0;
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for job in jobs {
            handles.push(scope.spawn(move || {
                if *job < 0 {
                    panic!("negative job");
                }
                *job
            }));
        }
        for handle in handles {
            if let Ok(value) = handle.join() {
                total += value;
            }
        }
    });
    Ok(total)
}
