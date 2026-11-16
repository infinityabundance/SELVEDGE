//! First segment length.

pub fn first_segment_length(text: &str) -> usize {
    let segments: Vec<&str> = text.split(',').collect();
    segments.first().map(|segment| segment.len()).unwrap_or(0)
}
