//! Measure, version 2. The field `value` is now `amount`.

pub struct Measure {
    pub amount: f64,
}

pub fn load(text: &str) -> Option<Measure> {
    text.trim().parse().ok().map(|amount| Measure { amount })
}
