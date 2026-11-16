//! Matrix transpose.

pub fn transpose(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let rows = matrix.len();
    let columns = matrix.first().map(|row| row.len()).unwrap_or(0);
    let mut out = vec![Vec::with_capacity(rows); columns];
    for row in matrix {
        for (column, value) in row.iter().enumerate() {
            out[column].push(*value);
        }
    }
    out
}
