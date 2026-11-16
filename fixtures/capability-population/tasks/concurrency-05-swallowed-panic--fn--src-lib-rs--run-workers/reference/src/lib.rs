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

#[cfg(test)]
mod visible {
    use crate::run_workers;

    #[test]
    fn sums_successful_workers() {
        assert_eq!(run_workers(&[1, 2, 3]), Ok(6));
    }

    #[test]
    fn a_panicking_worker_is_an_error() {
        assert_eq!(run_workers(&[-1]), Err("worker failed".to_string()));
    }

    #[test]
    fn a_mixed_batch_is_an_error() {
        assert_eq!(run_workers(&[1, -1, 2]), Err("worker failed".to_string()));
    }

    #[test]
    fn no_jobs_is_zero() {
        assert_eq!(run_workers(&[]), Ok(0));
    }
}
