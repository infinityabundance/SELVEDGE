use concurrency_01_lock_order_deadlock::run_pair;

#[test]
fn two_transfers_apply() {
    assert_eq!(run_pair(100, 50, 10), (80, 70));
}

#[test]
fn amounts_are_signed() {
    assert_eq!(run_pair(0, 100, -5), (10, 90));
}

#[test]
fn the_helpers_compose() {
    use std::sync::Mutex;
    let a = Mutex::new(10);
    let b = Mutex::new(0);
    concurrency_01_lock_order_deadlock::transfer_a_to_b(&a, &b, 3);
    concurrency_01_lock_order_deadlock::transfer_b_to_a(&b, &a, 1);
    assert_eq!(*a.lock().unwrap(), 8);
    assert_eq!(*b.lock().unwrap(), 2);
}
