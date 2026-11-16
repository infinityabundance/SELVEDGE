//! A graph whose degree-sum invariant is not preserved on removal.

#[derive(Debug, PartialEq)]
pub enum GraphError {
    OutOfRange,
    SelfLoop,
    Missing,
}

#[derive(Debug)]
pub struct Graph {
    nodes: usize,
    edges: Vec<(u32, u32)>,
    degrees: Vec<u32>,
}

impl Graph {
    pub fn new(nodes: usize) -> Self {
        Graph { nodes, edges: Vec::new(), degrees: vec![0; nodes] }
    }

    pub fn add_edge(&mut self, from: u32, to: u32) -> Result<(), GraphError> {
        if from as usize >= self.nodes || to as usize >= self.nodes {
            return Err(GraphError::OutOfRange);
        }
        self.edges.push((from, to));
        self.degrees[from as usize] += 1;
        self.degrees[to as usize] += 1;
        Ok(())
    }

    pub fn remove_edge(&mut self, from: u32, to: u32) -> Result<(), GraphError> {
        if let Some(position) = self.edges.iter().position(|edge| *edge == (from, to)) {
            self.edges.remove(position);
            Ok(())
        } else {
            Err(GraphError::Missing)
        }
    }

    pub fn degree(&self, node: u32) -> u32 {
        self.degrees[node as usize]
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn conserved(&self) -> bool {
        self.degrees.iter().sum::<u32>() as usize == 2 * self.edges.len()
    }
}
