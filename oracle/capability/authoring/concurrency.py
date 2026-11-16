"""Concurrency defects: deadlock, lost update, and a thread-safety boundary.

The deadlock is deterministic: each transfer helper sleeps while holding its
first lock, so two concurrent calls in opposite orders deadlock even on a
single core. The timeout is the recorded outcome for the failing state — a
failure, never a hang. The lost-update task is deterministic too: both threads
load, meet at a barrier, then store, so exactly one update is lost.

The hidden tests use a bounded wait or a barrier, so the reference solution's
runtime is milliseconds and the initial state's failure is reproducible.
"""

TASKS = [
    {
        "id": "concurrency-01-lock-order-deadlock",
        "category": "concurrency",
        "capability": "Remove a lock-order deadlock by imposing one global acquisition order.",
        "prompt_intro": (
            "`run_pair` deadlocks. `transfer_a_to_b` and `transfer_b_to_a` each sleep\n"
            "while holding their first lock, so two concurrent calls in opposite orders\n"
            "block forever. Impose one acquisition order (for example, by comparing the\n"
            "addresses of the two mutexes) so both helpers acquire in the same order\n"
            "regardless of the direction of the transfer. Keep the public functions and\n"
            "their effect: `run_pair(x, y, amount)` performs two a-to-b transfers and\n"
            "returns the final balances."),
        "observed_failure": (
            "test tests::two_transfers_apply has been running for over 60 seconds\n"
            "no test result: the process is blocked in `Mutex::lock` on both threads"),
        "limits": {"wall_seconds": 25.0, "cpu_seconds": 40},
        "files": {
            "src/lib.rs": (
                "//! Two accounts and two transfer helpers. Each helper sleeps while\n"
                "//! holding its first lock, so opposite orders deadlock.\n"
                "\n"
                "use std::sync::{Barrier, Mutex};\n"
                "use std::time::Duration;\n"
                "\n"
                "pub fn transfer_a_to_b(a: &Mutex<i64>, b: &Mutex<i64>, amount: i64) {\n"
                "    let mut guard_a = a.lock().unwrap();\n"
                "    std::thread::sleep(Duration::from_millis(10));\n"
                "    let mut guard_b = b.lock().unwrap();\n"
                "    *guard_a -= amount;\n"
                "    *guard_b += amount;\n"
                "}\n"
                "\n"
                "pub fn transfer_b_to_a(b: &Mutex<i64>, a: &Mutex<i64>, amount: i64) {\n"
                "    let mut guard_b = b.lock().unwrap();\n"
                "    std::thread::sleep(Duration::from_millis(10));\n"
                "    let mut guard_a = a.lock().unwrap();\n"
                "    *guard_a += amount;\n"
                "    *guard_b -= amount;\n"
                "}\n"
                "\n"
                "pub fn run_pair(x: i64, y: i64, amount: i64) -> (i64, i64) {\n"
                "    let a = Mutex::new(x);\n"
                "    let b = Mutex::new(y);\n"
                "    let barrier = Barrier::new(2);\n"
                "    std::thread::scope(|scope| {\n"
                "        scope.spawn(|| {\n"
                "            barrier.wait();\n"
                "            transfer_a_to_b(&a, &b, amount);\n"
                "        });\n"
                "        scope.spawn(|| {\n"
                "            barrier.wait();\n"
                "            transfer_b_to_a(&b, &a, -amount);\n"
                "        });\n"
                "    });\n"
                "    (a.into_inner().unwrap(), b.into_inner().unwrap())\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use concurrency_01_lock_order_deadlock::run_pair;\n"
                "\n"
                "#[test]\n"
                "fn two_transfers_apply() {\n"
                "    assert_eq!(run_pair(100, 50, 10), (80, 70));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn amounts_are_signed() {\n"
                "    assert_eq!(run_pair(0, 100, -5), (10, 90));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_helpers_compose() {\n"
                "    use std::sync::Mutex;\n"
                "    let a = Mutex::new(10);\n"
                "    let b = Mutex::new(0);\n"
                "    concurrency_01_lock_order_deadlock::transfer_a_to_b(&a, &b, 3);\n"
                "    concurrency_01_lock_order_deadlock::transfer_b_to_a(&b, &a, 1);\n"
                "    assert_eq!(*a.lock().unwrap(), 8);\n"
                "    assert_eq!(*b.lock().unwrap(), 2);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Two accounts with one global lock order.\n"
                "\n"
                "use std::sync::{Barrier, Mutex, MutexGuard};\n"
                "\n"
                "fn lock_pair<'a>(\n"
                "    a: &'a Mutex<i64>,\n"
                "    b: &'a Mutex<i64>,\n"
                ") -> (MutexGuard<'a, i64>, MutexGuard<'a, i64>) {\n"
                "    if (a as *const Mutex<i64>) < (b as *const Mutex<i64>) {\n"
                "        let guard_a = a.lock().unwrap();\n"
                "        let guard_b = b.lock().unwrap();\n"
                "        (guard_a, guard_b)\n"
                "    } else {\n"
                "        let guard_b = b.lock().unwrap();\n"
                "        let guard_a = a.lock().unwrap();\n"
                "        (guard_a, guard_b)\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn transfer_a_to_b(a: &Mutex<i64>, b: &Mutex<i64>, amount: i64) {\n"
                "    let (mut guard_a, mut guard_b) = lock_pair(a, b);\n"
                "    *guard_a -= amount;\n"
                "    *guard_b += amount;\n"
                "}\n"
                "\n"
                "pub fn transfer_b_to_a(b: &Mutex<i64>, a: &Mutex<i64>, amount: i64) {\n"
                "    let (mut guard_a, mut guard_b) = lock_pair(a, b);\n"
                "    *guard_a += amount;\n"
                "    *guard_b -= amount;\n"
                "}\n"
                "\n"
                "pub fn run_pair(x: i64, y: i64, amount: i64) -> (i64, i64) {\n"
                "    let a = Mutex::new(x);\n"
                "    let b = Mutex::new(y);\n"
                "    let barrier = Barrier::new(2);\n"
                "    std::thread::scope(|scope| {\n"
                "        scope.spawn(|| {\n"
                "            barrier.wait();\n"
                "            transfer_a_to_b(&a, &b, amount);\n"
                "        });\n"
                "        scope.spawn(|| {\n"
                "            barrier.wait();\n"
                "            transfer_b_to_a(&b, &a, -amount);\n"
                "        });\n"
                "    });\n"
                "    (a.into_inner().unwrap(), b.into_inner().unwrap())\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "*guard_a -= amount;", "replace": "*guard_a += amount;"},
        "initial_failure": {"kind": "timeout", "must_contain": ["running 3 tests"]},
    },
    {
        "id": "concurrency-02-lost-update",
        "category": "concurrency",
        "capability": "Replace a non-atomic read-modify-write with an atomic fetch-and-add.",
        "prompt_intro": (
            "`bump_pair` loses an update: both threads load the counter, meet at the\n"
            "barrier, and store the same value plus one, so the counter ends at 1\n"
            "instead of 2. `bump_all` has the same defect at scale. Use an atomic\n"
            "read-modify-write so every call adds exactly one. Keep both signatures and\n"
            "the barriers."),
        "observed_failure": (
            "thread 'tests::pair_bumps_twice' panicked at tests/check.rs:6:5:\n"
            "assertion `left == right` failed\n"
            "  left: 1\n"
            " right: 2"),
        "files": {
            "src/lib.rs": (
                "//! A counter incremented from two threads.\n"
                "\n"
                "use std::sync::atomic::{AtomicUsize, Ordering};\n"
                "use std::sync::Barrier;\n"
                "\n"
                "pub fn bump_pair(counter: &AtomicUsize) {\n"
                "    let barrier = Barrier::new(2);\n"
                "    std::thread::scope(|scope| {\n"
                "        for _ in 0..2 {\n"
                "            scope.spawn(|| {\n"
                "                let value = counter.load(Ordering::SeqCst);\n"
                "                barrier.wait();\n"
                "                counter.store(value + 1, Ordering::SeqCst);\n"
                "            });\n"
                "        }\n"
                "    });\n"
                "}\n"
                "\n"
                "pub fn bump_all(counter: &AtomicUsize, threads: usize, per_thread: usize) {\n"
                "    std::thread::scope(|scope| {\n"
                "        for _ in 0..threads {\n"
                "            scope.spawn(|| {\n"
                "                for _ in 0..per_thread {\n"
                "                    let value = counter.load(Ordering::Relaxed);\n"
                "                    std::hint::spin_loop();\n"
                "                    counter.store(value + 1, Ordering::Relaxed);\n"
                "                }\n"
                "            });\n"
                "        }\n"
                "    });\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use std::sync::atomic::{AtomicUsize, Ordering};\n"
                "\n"
                "use concurrency_02_lost_update::{bump_all, bump_pair};\n"
                "\n"
                "#[test]\n"
                "fn pair_bumps_twice() {\n"
                "    let counter = AtomicUsize::new(0);\n"
                "    bump_pair(&counter);\n"
                "    assert_eq!(counter.load(Ordering::SeqCst), 2);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn many_bumps_are_not_lost() {\n"
                "    let counter = AtomicUsize::new(0);\n"
                "    bump_all(&counter, 4, 500);\n"
                "    assert_eq!(counter.load(Ordering::SeqCst), 2000);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn bumps_start_from_the_current_value() {\n"
                "    let counter = AtomicUsize::new(41);\n"
                "    bump_pair(&counter);\n"
                "    assert_eq!(counter.load(Ordering::SeqCst), 43);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! A counter incremented atomically from two threads.\n"
                "\n"
                "use std::sync::atomic::{AtomicUsize, Ordering};\n"
                "use std::sync::Barrier;\n"
                "\n"
                "pub fn bump_pair(counter: &AtomicUsize) {\n"
                "    let barrier = Barrier::new(2);\n"
                "    std::thread::scope(|scope| {\n"
                "        for _ in 0..2 {\n"
                "            scope.spawn(|| {\n"
                "                barrier.wait();\n"
                "                counter.fetch_add(1, Ordering::SeqCst);\n"
                "            });\n"
                "        }\n"
                "    });\n"
                "}\n"
                "\n"
                "pub fn bump_all(counter: &AtomicUsize, threads: usize, per_thread: usize) {\n"
                "    std::thread::scope(|scope| {\n"
                "        for _ in 0..threads {\n"
                "            scope.spawn(|| {\n"
                "                for _ in 0..per_thread {\n"
                "                    counter.fetch_add(1, Ordering::Relaxed);\n"
                "                }\n"
                "            });\n"
                "        }\n"
                "    });\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs", "find": "counter.fetch_add(1, Ordering::SeqCst)",
                     "replace": "counter.fetch_add(2, Ordering::SeqCst)"},
        "initial_failure": {"kind": "test_failure",
                            "must_contain": ["assertion `left == right` failed"]},
    },
    {
        "id": "concurrency-03-send-boundary",
        "category": "concurrency",
        "capability": "Recognise a non-Send smart pointer crossing a thread boundary and move to Arc.",
        "prompt_intro": (
            "This crate does not build: `Rc<Vec<i64>>` is not `Send`, so the spawned\n"
            "worker cannot capture it. Use a thread-safe shared pointer and keep the\n"
            "result identical (the sum of every element, computed in parallel chunks).\n"
            "Keep the signature."),
        "observed_failure": (
            "error[E0277]: `Rc<Vec<i64>>` cannot be sent between threads safely\n"
            " --> src/lib.rs:10:13\n"
            "  |\n"
            "10|         let part = Rc::clone(&shared);\n"
            "  |             ^^^^^^^^^^^^^^^^^^^^^^^^^ `Rc<Vec<i64>>` is not `Send`"),
        "files": {
            "src/lib.rs": (
                "//! Sum a vector across four worker threads.\n"
                "\n"
                "use std::rc::Rc;\n"
                "\n"
                "pub fn total_in_threads(data: Vec<i64>) -> i64 {\n"
                "    if data.is_empty() {\n"
                "        return 0;\n"
                "    }\n"
                "    let shared = Rc::new(data);\n"
                "    let workers = 4.min(shared.len());\n"
                "    let chunk = shared.len().div_ceil(workers);\n"
                "    let handles: Vec<_> = (0..workers)\n"
                "        .map(|index| {\n"
                "            let part = Rc::clone(&shared);\n"
                "            std::thread::spawn(move || {\n"
                "                let start = (index * chunk).min(part.len());\n"
                "                let end = ((index + 1) * chunk).min(part.len());\n"
                "                part[start..end].iter().copied().sum::<i64>()\n"
                "            })\n"
                "        })\n"
                "        .collect();\n"
                "    handles\n"
                "        .into_iter()\n"
                "        .map(|handle| handle.join().unwrap())\n"
                "        .sum()\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use concurrency_03_send_boundary::total_in_threads;\n"
                "\n"
                "#[test]\n"
                "fn sums_every_element() {\n"
                "    assert_eq!(total_in_threads(vec![1, 2, 3, 4, 5]), 15);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn empty_is_zero() {\n"
                "    assert_eq!(total_in_threads(vec![]), 0);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn one_element() {\n"
                "    assert_eq!(total_in_threads(vec![7]), 7);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn negative_values() {\n"
                "    assert_eq!(total_in_threads(vec![-4, 1, 3]), 0);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Sum a vector across worker threads with a thread-safe pointer.\n"
                "\n"
                "use std::sync::Arc;\n"
                "\n"
                "pub fn total_in_threads(data: Vec<i64>) -> i64 {\n"
                "    if data.is_empty() {\n"
                "        return 0;\n"
                "    }\n"
                "    let shared = Arc::new(data);\n"
                "    let workers = 4.min(shared.len());\n"
                "    let chunk = shared.len().div_ceil(workers);\n"
                "    let handles: Vec<_> = (0..workers)\n"
                "        .map(|index| {\n"
                "            let part = Arc::clone(&shared);\n"
                "            std::thread::spawn(move || {\n"
                "                let start = (index * chunk).min(part.len());\n"
                "                let end = ((index + 1) * chunk).min(part.len());\n"
                "                part[start..end].iter().copied().sum::<i64>()\n"
                "            })\n"
                "        })\n"
                "        .collect();\n"
                "    handles\n"
                "        .into_iter()\n"
                "        .map(|handle| handle.join().unwrap())\n"
                "        .sum()\n"
                "}\n"),
        },
        "mutation": {"file": "src/lib.rs",
                     "find": "part[start..end].iter().copied().sum::<i64>()",
                     "replace": "part[start..end].iter().copied().take(1).sum::<i64>()"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
]
