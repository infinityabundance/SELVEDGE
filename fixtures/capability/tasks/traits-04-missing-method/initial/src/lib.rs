//! Shapes. `Square` does not implement the full trait.

pub trait Shape {
    fn area(&self) -> f64;
    fn name(&self) -> &'static str;
}

pub struct Square {
    pub side: f64,
}

impl Shape for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
}

pub fn describe<T: Shape>(shape: &T) -> String {
    format!("{}:{:.1}", shape.name(), shape.area())
}
