//! A codec checksum.

pub fn codec_total(text: &str) -> u32 {
    codec_impl::total(text)
}
