//! Nodes that can be debug-formatted.

use core::fmt::Debug;

pub trait Node: Debug {
    fn id(&self) -> u32;
}

pub struct Leaf {
    pub id: u32,
}

impl Node for Leaf {
    fn id(&self) -> u32 {
        self.id
    }
}

pub fn summarise(nodes: &[Box<dyn Node>]) -> String {
    nodes
        .iter()
        .map(|node| format!("{:?}:{}", node, node.id()))
        .collect::<Vec<_>>()
        .join(",")
}
