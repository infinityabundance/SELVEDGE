//! Codec, version 2. The library target is `codec_impl` while the
//! package is `codec`.

pub fn total(text: &str) -> u32 {
    text.bytes().map(|byte| byte as u32).sum()
}
