//! Scoped workers that propagate a panic.

use std::thread;

pub fn run_workers(jobs: &[i32]) -> Result<i32, String> {
    let mut total = 0;
    let result = thread::scope(|scope| -> Result<(), String> {
        let handles: Vec<_> = jobs
            .iter()
            .map(|job| {
                scope.spawn(move || {
                    if *job < 0 {
                        panic!("negative job");
                    }
                    *job
                })
            })
            .collect();
        for handle in handles {
            total += handle.join().map_err(|_| "worker failed".to_string())?;
        }
        Ok(())
    });
    result?;
    Ok(total)
}
