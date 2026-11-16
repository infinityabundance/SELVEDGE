"""Concurrency bugs, batch two (six tasks).

The set keeps the outcome classes distinct: two compile failures (a non-`Send`
guard and borrowed data escaping a thread), two deterministic lost-update /
swallowed-panic failures, and two channel ownership errors. No task relies on a
45-second deadlock to fail.
"""

from .common import code

TASKS = [
    {
        "id": "concurrency-04-deterministic-lost-update",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Replace a non-atomic load-and-store pair with an atomic read-modify-write.",
        "prompt_intro": (
            "The hidden test fails. Two threads load a counter, synchronise on a "
            "barrier, and store `loaded + 1`. Both increments must be counted. "
            "Rewrite `increment_pair` using an atomic read-modify-write and keep the "
            "barrier so the race is still exercised. Keep the signature "
            "`pub fn increment_pair() -> usize`."),
        "observed_failure": (
            "thread 'both_increments_are_counted' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: 1\n"
            " right: 2"),
        "files": {
            "src/lib.rs": code("""
                //! Two synchronized increments that lose one update.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::{Arc, Barrier};
                use std::thread;

                pub fn increment_pair() -> usize {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let barrier = Arc::new(Barrier::new(2));
                    let handles: Vec<_> = (0..2)
                        .map(|_| {
                            let counter = Arc::clone(&counter);
                            let barrier = Arc::clone(&barrier);
                            thread::spawn(move || {
                                let loaded = counter.load(Ordering::SeqCst);
                                barrier.wait();
                                counter.store(loaded + 1, Ordering::SeqCst);
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    counter.load(Ordering::SeqCst)
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_04_deterministic_lost_update::increment_pair;

                #[test]
                fn both_increments_are_counted() {
                    assert_eq!(increment_pair(), 2);
                }

                #[test]
                fn the_result_is_stable_across_runs() {
                    for _ in 0..20 {
                        assert_eq!(increment_pair(), 2);
                    }
                }

                #[test]
                fn the_counter_is_not_one() {
                    assert_ne!(increment_pair(), 1);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Two synchronized increments.

                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::{Arc, Barrier};
                use std::thread;

                pub fn increment_pair() -> usize {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let barrier = Arc::new(Barrier::new(2));
                    let handles: Vec<_> = (0..2)
                        .map(|_| {
                            let counter = Arc::clone(&counter);
                            let barrier = Arc::clone(&barrier);
                            thread::spawn(move || {
                                barrier.wait();
                                counter.fetch_add(1, Ordering::SeqCst);
                            })
                        })
                        .collect();
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    counter.load(Ordering::SeqCst)
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.fetch_add(1, Ordering::SeqCst)",
                     "replace": "counter.store(1, Ordering::SeqCst)"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "concurrency-05-swallowed-panic",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Propagate a worker panic instead of discarding the join error.",
        "prompt_intro": (
            "The hidden test fails. `run_workers` joins scoped workers and silently "
            "ignores a panicking one. It must return `Err(\"worker failed\")` if any "
            "worker panics, and otherwise the sum of the job values. Keep the "
            "signature `pub fn run_workers(jobs: &[i32]) -> Result<i32, String>`."),
        "observed_failure": (
            "thread 'a_panicking_worker_is_an_error' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: Ok(0)\n"
            " right: Err(\"worker failed\")"),
        "files": {
            "src/lib.rs": code("""
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
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_05_swallowed_panic::run_workers;

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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
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
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '"worker failed"', "replace": '"worker"'},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "concurrency-06-guard-crosses-thread",
        "category": "concurrency",
        "difficulty": "medium",
        "capability": "Move the lock acquisition into the spawned thread instead of moving the guard.",
        "prompt_intro": (
            "This crate does not compile. `push_from_thread` moves a `MutexGuard` "
            "into a spawned thread; guards are not `Send`. Acquire the lock inside "
            "the thread instead. Keep the signature "
            "`pub fn push_from_thread(values: &Arc<Mutex<Vec<i32>>>, value: i32)`."),
        "observed_failure": (
            "error[E0277]: `MutexGuard<'_, Vec<i32>>` cannot be sent between threads safely\n"
            " --> src/lib.rs:6:19\n"
            "  |\n"
            "6 |     thread::spawn(move || {\n"
            "  |                   ^^^^^^^ `MutexGuard<'_, Vec<i32>>` cannot be sent between threads safely"),
        "files": {
            "src/lib.rs": code("""
                //! Push from a spawned thread.

                use std::sync::{Arc, Mutex};
                use std::thread;

                pub fn push_from_thread(values: &Arc<Mutex<Vec<i32>>>, value: i32) {
                    let mut guard = values.lock().unwrap();
                    thread::spawn(move || {
                        guard.push(value);
                    })
                    .join()
                    .unwrap();
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_06_guard_crosses_thread::push_from_thread;
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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
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
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "push(value)", "replace": "push(value + 1)"},
        "initial_failure": {"kind": "compile_error",
                            "must_contain": ["cannot be sent between threads safely"]},
    },
    {
        "id": "concurrency-07-silent-channel-drop",
        "category": "concurrency",
        "difficulty": "easy",
        "capability": "Report channel saturation instead of silently dropping values.",
        "prompt_intro": (
            "The hidden test fails. `send_all` currently ignores every `try_send` "
            "error, so a full or disconnected channel loses values silently. It must "
            "return `Err(\"channel full\")` for a full channel and "
            "`Err(\"disconnected\")` for a disconnected one; otherwise send every "
            "value and return `Ok(())`. Keep the signature "
            "`pub fn send_all(sender: &SyncSender<i32>, values: &[i32]) -> Result<(), String>`."),
        "observed_failure": (
            "thread 'a_full_channel_is_an_error' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: Ok(())\n"
            " right: Err(\"channel full\")"),
        "files": {
            "src/lib.rs": code("""
                //! Send every value, currently discarding every error.

                use std::sync::mpsc::SyncSender;

                pub fn send_all(sender: &SyncSender<i32>, values: &[i32]) -> Result<(), String> {
                    for value in values {
                        let _ = sender.try_send(*value);
                    }
                    Ok(())
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_07_silent_channel_drop::send_all;
                use std::sync::mpsc::sync_channel;

                #[test]
                fn sends_when_there_is_room() {
                    let (sender, receiver) = sync_channel(2);
                    assert_eq!(send_all(&sender, &[1, 2]), Ok(()));
                    assert_eq!(receiver.recv(), Ok(1));
                    assert_eq!(receiver.recv(), Ok(2));
                }

                #[test]
                fn a_full_channel_is_an_error() {
                    let (sender, _receiver) = sync_channel(1);
                    sender.try_send(9).unwrap();
                    assert_eq!(send_all(&sender, &[1]), Err("channel full".to_string()));
                }

                #[test]
                fn a_disconnected_channel_is_an_error() {
                    let (sender, receiver) = sync_channel(1);
                    drop(receiver);
                    assert_eq!(send_all(&sender, &[1]), Err("disconnected".to_string()));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Send every value and report saturation.

                use std::sync::mpsc::{SyncSender, TrySendError};

                pub fn send_all(sender: &SyncSender<i32>, values: &[i32]) -> Result<(), String> {
                    for value in values {
                        match sender.try_send(*value) {
                            Ok(()) => {}
                            Err(TrySendError::Full(_)) => {
                                return Err("channel full".to_string())
                            }
                            Err(TrySendError::Disconnected(_)) => {
                                return Err("disconnected".to_string())
                            }
                        }
                    }
                    Ok(())
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '"channel full"', "replace": '"full"'},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "concurrency-08-borrowed-thread",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Use scoped threads for borrows that cannot satisfy a 'static spawn.",
        "prompt_intro": (
            "This crate does not compile. `sum_parallel` sends slices of its input "
            "to `thread::spawn`, which requires a `'static` borrow. Use a scoped "
            "thread instead. Keep the signature "
            "`pub fn sum_parallel(values: &[i32]) -> i32` and the two-element chunks."),
        "observed_failure": (
            "error[E0521]: borrowed data escapes outside of function\n"
            " --> src/lib.rs:4:56\n"
            "  |\n"
            "3 |     let handles: Vec<_> = values\n"
            "  |                           ------ `values` is a reference that is only valid in the function body\n"
            "4 |         .chunks(2)\n"
            "  |          ^^^^^^^\n"
            "  |          `values` escapes the function body here"),
        "files": {
            "src/lib.rs": code("""
                //! Sum two-element chunks in parallel.

                use std::thread;

                pub fn sum_parallel(values: &[i32]) -> i32 {
                    let handles: Vec<_> = values
                        .chunks(2)
                        .map(|chunk| thread::spawn(move || chunk.iter().sum::<i32>()))
                        .collect();
                    handles.into_iter().map(|handle| handle.join().unwrap()).sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_08_borrowed_thread::sum_parallel;

                #[test]
                fn sums_every_value() {
                    assert_eq!(sum_parallel(&[1, 2, 3, 4, 5]), 15);
                }

                #[test]
                fn an_empty_slice_is_zero() {
                    assert_eq!(sum_parallel(&[]), 0);
                }

                #[test]
                fn a_single_value_is_itself() {
                    assert_eq!(sum_parallel(&[7]), 7);
                }

                #[test]
                fn negative_values_are_summed() {
                    assert_eq!(sum_parallel(&[-3, 4, -5]), -4);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Sum two-element chunks in parallel.

                use std::thread;

                pub fn sum_parallel(values: &[i32]) -> i32 {
                    thread::scope(|scope| {
                        let handles: Vec<_> = values
                            .chunks(2)
                            .map(|chunk| scope.spawn(move || chunk.iter().sum::<i32>()))
                            .collect();
                        handles.into_iter().map(|handle| handle.join().unwrap()).sum()
                    })
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "chunk.iter().sum::<i32>()",
                     "replace": "chunk.iter().count() as i32"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0521"]},
    },
    {
        "id": "concurrency-09-shared-input",
        "category": "concurrency",
        "difficulty": "hard",
        "capability": "Share an owned input between two threads with Arc instead of moving it twice.",
        "prompt_intro": (
            "This crate does not compile. `square_sum` moves the owned vector into "
            "the first worker and then tries to move it into the second. Share it "
            "with `Arc` so both workers read the same input, and return their common "
            "result. Keep the signature `pub fn square_sum(values: Vec<i32>) -> i32`."),
        "observed_failure": (
            "error[E0382]: use of moved value: `values`\n"
            " --> src/lib.rs:5:44\n"
            "  |\n"
            "3 |     let first = thread::spawn(move || values.iter().map(|v| v * v).sum::<i32>());\n"
            "  |                                         ------ value moved into closure here\n"
            "4 |\n"
            "5 |     let second = thread::spawn(move || values.iter().map(|v| v * v).sum::<i32>());\n"
            "  |                                            ^^^^^^ value used here after move"),
        "files": {
            "src/lib.rs": code("""
                //! Sum of squares, computed twice by two workers.

                use std::thread;

                pub fn square_sum(values: Vec<i32>) -> i32 {
                    let first = thread::spawn(move || {
                        values.iter().map(|value| value * value).sum::<i32>()
                    });
                    let second = thread::spawn(move || {
                        values.iter().map(|value| value * value).sum::<i32>()
                    });
                    let left = first.join().unwrap();
                    let right = second.join().unwrap();
                    assert_eq!(left, right);
                    left
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use concurrency_09_shared_input::square_sum;

                #[test]
                fn sums_squares() {
                    assert_eq!(square_sum(vec![1, 2, 3]), 14);
                }

                #[test]
                fn empty_input_is_zero() {
                    assert_eq!(square_sum(Vec::new()), 0);
                }

                #[test]
                fn negative_values_square_positively() {
                    assert_eq!(square_sum(vec![-2, 3]), 13);
                }

                #[test]
                fn zeroes_contribute_nothing() {
                    assert_eq!(square_sum(vec![0, 0, 2]), 4);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Sum of squares, computed twice by two workers.

                use std::sync::Arc;
                use std::thread;

                pub fn square_sum(values: Vec<i32>) -> i32 {
                    let values = Arc::new(values);
                    let first = {
                        let values = Arc::clone(&values);
                        thread::spawn(move || {
                            values.iter().map(|value| value * value).sum::<i32>()
                        })
                    };
                    let second = {
                        let values = Arc::clone(&values);
                        thread::spawn(move || {
                            values.iter().map(|value| value * value).sum::<i32>()
                        })
                    };
                    let left = first.join().unwrap();
                    let right = second.join().unwrap();
                    assert_eq!(left, right);
                    left
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "value * value", "replace": "value + value"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0382"]},
    },
]
