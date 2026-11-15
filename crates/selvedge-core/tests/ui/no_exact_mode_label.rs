// A decision labelled `SelectionMode::Exact` cannot be minted from outside the
// crate, and no exact-labelled constructor exists at all. The label is a
// declaration, not evidence; exactness is `ExactSelection`, which needs a
// `NativeSetWitness`.
//@ expect: no associated function or constant named `native`
use selvedge_core::SelectionDecision;

fn main() {
    let _ = SelectionDecision::native(Vec::new());
}
