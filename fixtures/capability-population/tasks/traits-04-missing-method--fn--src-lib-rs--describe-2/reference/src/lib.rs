//! Shapes.

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

    fn name(&self) -> &'static str {
        "square"
    }
}

pub fn describe<T: Shape>(shape: &T) -> String {
    format!("{}:{:.1}", shape.name(), shape.area())
}

#[cfg(test)]
mod visible {
    use crate::{describe, Shape, Square};

    #[test]
    fn reports_the_area() {
        assert_eq!(Square { side: 3.0 }.area(), 9.0);
    }

    #[test]
    fn reports_the_name() {
        assert_eq!(Square { side: 3.0 }.name(), "square");
    }

    #[test]
    fn describes_a_shape() {
        assert_eq!(describe(&Square { side: 2.0 }), "square:4.0");
    }

    #[test]
    fn works_with_another_implementor() {
        struct Triangle;
        impl Shape for Triangle {
            fn area(&self) -> f64 { 6.0 }
            fn name(&self) -> &'static str { "triangle" }
        }
        assert_eq!(describe(&Triangle), "triangle:6.0");
    }
}
