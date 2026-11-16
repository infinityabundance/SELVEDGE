use sealed_failing_03_rectangle_transpose::transpose;

#[test]
fn a_two_by_three_matrix_transposes() {
    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
    assert_eq!(transpose(&matrix),
               vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
}

#[test]
fn a_square_matrix_transposes() {
    let matrix = vec![vec![1, 2], vec![3, 4]];
    assert_eq!(transpose(&matrix), vec![vec![1, 3], vec![2, 4]]);
}

#[test]
fn an_empty_matrix_stays_empty() {
    assert!(transpose(&[]).is_empty());
}

#[test]
fn a_single_row_transposes_to_a_column() {
    let matrix = vec![vec![1, 2, 3]];
    assert_eq!(transpose(&matrix), vec![vec![1], vec![2], vec![3]]);
}
