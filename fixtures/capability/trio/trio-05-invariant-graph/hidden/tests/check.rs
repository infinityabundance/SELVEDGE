use trio_05_invariant_graph::{Graph, GraphError};

#[test]
fn add_updates_both_degrees() {
    let mut graph = Graph::new(3);
    graph.add_edge(0, 1).unwrap();
    assert_eq!(graph.degree(0), 1);
    assert_eq!(graph.degree(1), 1);
    assert!(graph.conserved());
}

#[test]
fn remove_updates_both_degrees() {
    let mut graph = Graph::new(3);
    graph.add_edge(0, 1).unwrap();
    graph.remove_edge(0, 1).unwrap();
    assert_eq!(graph.degree(0), 0);
    assert_eq!(graph.degree(1), 0);
    assert_eq!(graph.edge_count(), 0);
    assert!(graph.conserved());
}

#[test]
fn a_self_loop_is_rejected_without_a_partial_update() {
    let mut graph = Graph::new(2);
    assert_eq!(graph.add_edge(1, 1), Err(GraphError::SelfLoop));
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.degree(1), 0);
    assert!(graph.conserved());
}

#[test]
fn a_missing_edge_is_rejected_without_a_partial_update() {
    let mut graph = Graph::new(2);
    assert_eq!(graph.remove_edge(0, 1), Err(GraphError::Missing));
    assert!(graph.conserved());
}

#[test]
fn conservation_holds_after_a_mixed_sequence() {
    let mut graph = Graph::new(3);
    graph.add_edge(0, 1).unwrap();
    graph.add_edge(1, 2).unwrap();
    graph.remove_edge(0, 1).unwrap();
    assert_eq!(graph.degree(1), 1);
    assert_eq!(graph.degree(2), 1);
    assert!(graph.conserved());
}
