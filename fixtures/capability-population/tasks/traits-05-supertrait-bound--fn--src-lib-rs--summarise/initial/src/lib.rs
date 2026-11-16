//! Nodes that can be debug-formatted.

use core::fmt::Debug;

pub trait Node: Debug {
    fn id(&self) -> u32;
}

#[derive(Debug)]
pub struct Leaf {
    pub id: u32,
}

impl Node for Leaf {
    fn id(&self) -> u32 {
        self.id
    }
}

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{summarise, Leaf, Node};

    #[test]
    fn reports_the_id() {
        assert_eq!(Leaf { id: 2 }.id(), 2);
    }

    #[test]
    fn leaf_is_debug() {
        assert!(format!("{:?}", Leaf { id: 7 }).contains("Leaf"));
    }

    #[test]
    fn summarises_a_boxed_node() {
        let nodes: Vec<Box<dyn Node>> = vec![Box::new(Leaf { id: 4 })];
        let text = summarise(&nodes);
        assert!(text.contains("4"));
        assert!(text.contains("Leaf"));
    }

    #[test]
    fn summarises_several_nodes() {
        let nodes: Vec<Box<dyn Node>> = vec![
            Box::new(Leaf { id: 1 }),
            Box::new(Leaf { id: 2 }),
        ];
        assert_eq!(summarise(&nodes).matches("Leaf").count(), 2);
    }
}
