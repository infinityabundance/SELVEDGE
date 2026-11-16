//! Traffic-light transitions. The visible test fails: Green skips
//! Yellow.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light {
    Red,
    Green,
    Yellow,
}

pub fn next(light: Light) -> Light {
    match light {
        Light::Red => Light::Green,
        Light::Green => Light::Red,
        Light::Yellow => Light::Green,
    }
}

#[cfg(test)]
mod tests {
    use super::{next, Light};

    #[test]
    fn green_follows_red_then_yellow() {
        assert_eq!(next(Light::Red), Light::Green);
        assert_eq!(next(Light::Green), Light::Yellow);
    }
}
