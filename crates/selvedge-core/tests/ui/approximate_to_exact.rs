// The approximate-to-exact escape hatch was removed. Neither constructor
// exists, so a selector cannot relabel its subset as the native computation.
//@ expect: no associated function or constant named `reclassify_from_approximate`
use selvedge_core::{ExactSelection, SelectionDecision, SelectionProvenance};

fn main() {
    let provenance = SelectionProvenance::approximate("predictor", Vec::new()).unwrap();
    let decision = SelectionDecision::new(Vec::new(), provenance).unwrap();
    let _ = ExactSelection::reclassify_from_approximate(decision, "verified externally");
}
