//! A codec fold. This does not build: the dependency path is wrong.

pub fn codec_fold(text: &str) -> u32 {
    codec::fold(text)
}
