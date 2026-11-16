//! First segment length. The crate does not build: the iterator borrows
//! a temporary.

pub fn first_segment_length(text: &str) -> usize {
    let mut segments = text.split(',').collect::<Vec<_>>().iter();
    segments.next().map(|segment| segment.len()).unwrap_or(0)
}
