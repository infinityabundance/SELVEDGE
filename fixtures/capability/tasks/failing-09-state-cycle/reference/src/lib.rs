//! Traffic-light transitions.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light {
    Red,
    Green,
    Yellow,
}

pub fn next(light: Light) -> Light {
    match light {
        Light::Red => Light::Green,
        Light::Green => Light::Yellow,
        Light::Yellow => Light::Red,
    }
}
