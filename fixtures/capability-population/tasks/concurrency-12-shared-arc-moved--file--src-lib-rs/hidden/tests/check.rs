use std::sync::Arc;

use concurrency_12_shared_arc_moved::chunked_sum;

#[test]
fn two_workers_cover_every_index() {
    let values = Arc::new(vec![1, 2, 3, 4]);
    assert_eq!(chunked_sum(values, 2), 10);
}

#[test]
fn one_worker_sums_everything() {
    let values = Arc::new(vec![5, 6, 7]);
    assert_eq!(chunked_sum(values, 1), 18);
}

#[test]
fn an_empty_vector_sums_to_zero() {
    let values = Arc::new(Vec::new());
    assert_eq!(chunked_sum(values, 3), 0);
}
