//! Matrix transpose assuming a square input.

pub fn transpose(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let size = matrix.len();
    let mut out = vec![vec![0i32; size]; size];
    for row in 0..size {
        for column in 0..size {
            out[column][row] = matrix[row][column];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::transpose;

    #[test]
    fn visible_rectangular() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(transpose(&matrix),
                   vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
    }
}
