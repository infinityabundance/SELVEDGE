"""Anti-specialisation trio, operations two and three (six tasks).

The first trio applies one operation (dependency-closure minimal repair) on
three surfaces. These six apply two further operations on the same three
surfaces, so every surface has three independently checked tasks:

* **invariant preservation** -- identify the state relation, apply a transition
  to every component that carries it, and reject (leaving state unchanged) any
  transition that would break it;
* **quantifier/scope resolution** -- resolve a name against the nearest binding,
  distinguish free from bound occurrences, and never leak an inner binding.

Surfaces are `rust`, `numeric-graph` and `logic`. The check stage aggregates by
surface across all three operation families; a transfer win requires all three
surfaces fully solved, and per-surface results are never pooled.
"""

from .common import code

_INVARIANT = {
    "name": "invariant preservation",
    "definition": (
        "Given a declared state invariant, apply a transition to every component "
        "that carries the invariant, reject a transition that would violate it and "
        "leave the state unchanged, and never let a partial update escape."),
    "transfer_rule": (
        "A surface is solved only when every task on that surface passes; the trio "
        "win requires all three surfaces and is never pooled across them."),
    "audit": "author annotation only; no inter-annotator agreement yet (P11.21 gap)",
}

_QUANTIFIER = {
    "name": "quantifier/scope resolution",
    "definition": (
        "Resolve each occurrence against the nearest enclosing binding for that "
        "name, distinguish free from bound occurrences, reject or reject-free an "
        "unbound name as the surface requires, and never leak an inner scope."),
    "transfer_rule": (
        "A surface is solved only when every task on that surface passes; the trio "
        "win requires all three surfaces and is never pooled across them."),
    "audit": "author annotation only; no inter-annotator agreement yet (P11.21 gap)",
}

TASKS = [
    {
        "id": "trio-04-invariant-ledger",
        "kind": "trio",
        "category": "trio",
        "capability": "Preserve a ledger invariant across deposits and withdrawals, rejecting overdraw.",
        "prompt_intro": (
            "The crate builds, but `Ledger` breaks its declared invariant "
            "`total == sum(entries)` after a withdrawal and allows an overdraw. "
            "Preserve the invariant on every transition: withdraw updates both "
            "components, rejects a non-positive amount with `invalid amount`, and "
            "rejects an overdraw with `insufficient funds` while leaving the ledger "
            "unchanged. Keep the public API."),
        "observed_failure": (
            "thread 'withdraw_updates_both_components' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: 10\n"
            " right: 6"),
        "shared_operation": dict(_INVARIANT, surface="rust"),
        "files": {
            "src/lib.rs": code("""
                //! A ledger whose invariant is not preserved by withdrawal.

                #[derive(Debug)]
                pub struct Ledger {
                    entries: Vec<i64>,
                    total: i64,
                }

                impl Ledger {
                    pub fn new() -> Self {
                        Ledger { entries: Vec::new(), total: 0 }
                    }

                    pub fn deposit(&mut self, amount: i64) {
                        self.entries.push(amount);
                        self.total += amount;
                    }

                    pub fn withdraw(&mut self, amount: i64) -> Result<(), String> {
                        if amount <= 0 {
                            return Err("invalid amount".to_string());
                        }
                        self.entries.push(-amount);
                        Ok(())
                    }

                    pub fn total(&self) -> i64 {
                        self.total
                    }

                    pub fn balance(&self) -> i64 {
                        self.entries.iter().sum()
                    }

                    pub fn entries(&self) -> &[i64] {
                        &self.entries
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use trio_04_invariant_ledger::Ledger;

                #[test]
                fn deposit_updates_both_components() {
                    let mut ledger = Ledger::new();
                    ledger.deposit(10);
                    assert_eq!(ledger.total(), 10);
                    assert_eq!(ledger.balance(), 10);
                }

                #[test]
                fn withdraw_updates_both_components() {
                    let mut ledger = Ledger::new();
                    ledger.deposit(10);
                    ledger.withdraw(4).unwrap();
                    assert_eq!(ledger.total(), 6);
                    assert_eq!(ledger.balance(), 6);
                    assert_eq!(ledger.entries(), &[10, -4]);
                }

                #[test]
                fn overdraw_is_rejected_and_leaves_state_unchanged() {
                    let mut ledger = Ledger::new();
                    ledger.deposit(5);
                    assert_eq!(ledger.withdraw(6), Err("insufficient funds".to_string()));
                    assert_eq!(ledger.total(), 5);
                    assert_eq!(ledger.balance(), 5);
                    assert_eq!(ledger.entries(), &[5]);
                }

                #[test]
                fn invalid_amount_is_rejected_and_leaves_state_unchanged() {
                    let mut ledger = Ledger::new();
                    ledger.deposit(5);
                    assert_eq!(ledger.withdraw(0), Err("invalid amount".to_string()));
                    assert_eq!(ledger.withdraw(-1), Err("invalid amount".to_string()));
                    assert_eq!(ledger.total(), 5);
                    assert_eq!(ledger.entries(), &[5]);
                }

                #[test]
                fn the_invariant_holds_after_a_sequence() {
                    let mut ledger = Ledger::new();
                    ledger.deposit(10);
                    ledger.withdraw(3).unwrap();
                    ledger.deposit(2);
                    assert_eq!(ledger.total(), ledger.balance());
                    assert_eq!(ledger.total(), 9);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A ledger that preserves its invariant.

                #[derive(Debug)]
                pub struct Ledger {
                    entries: Vec<i64>,
                    total: i64,
                }

                impl Ledger {
                    pub fn new() -> Self {
                        Ledger { entries: Vec::new(), total: 0 }
                    }

                    pub fn deposit(&mut self, amount: i64) {
                        self.entries.push(amount);
                        self.total += amount;
                    }

                    pub fn withdraw(&mut self, amount: i64) -> Result<(), String> {
                        if amount <= 0 {
                            return Err("invalid amount".to_string());
                        }
                        if amount > self.balance() {
                            return Err("insufficient funds".to_string());
                        }
                        self.entries.push(-amount);
                        self.total -= amount;
                        Ok(())
                    }

                    pub fn total(&self) -> i64 {
                        self.total
                    }

                    pub fn balance(&self) -> i64 {
                        self.entries.iter().sum()
                    }

                    pub fn entries(&self) -> &[i64] {
                        &self.entries
                    }
                }
            """),
        },
        "operation_items": [
            {"test": "deposit_updates_both_components",
             "operation": "apply the transition to every component carrying the invariant"},
            {"test": "withdraw_updates_both_components",
             "operation": "keep total and entries consistent after a partial-state transition"},
            {"test": "overdraw_is_rejected_and_leaves_state_unchanged",
             "operation": "reject a violating transition without a partial update"},
            {"test": "invalid_amount_is_rejected_and_leaves_state_unchanged",
             "operation": "precondition check before any component is mutated"},
            {"test": "the_invariant_holds_after_a_sequence",
             "operation": "the invariant is closed under a sequence of transitions"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "self.total -= amount;",
                     "replace": "self.total += amount;"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-05-invariant-graph",
        "kind": "trio",
        "category": "trio",
        "difficulty": "hard",
        "capability": "Preserve the degree-sum invariant across edge insertion and removal.",
        "prompt_intro": (
            "The crate builds, but `Graph` breaks its declared invariant "
            "`sum(degrees) == 2 * edges.len()` on removal and accepts self-loops. "
            "Preserve the invariant on every transition: `remove_edge` decrements "
            "both endpoint degrees, `add_edge` rejects a self-loop with `SelfLoop`, "
            "and an unknown edge or node is rejected without a partial update. Keep "
            "the public API."),
        "observed_failure": (
            "thread 'remove_updates_both_degrees' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: 1\n"
            " right: 0"),
        "shared_operation": dict(_INVARIANT, surface="numeric-graph"),
        "files": {
            "src/lib.rs": code("""
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
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! A graph that preserves its degree-sum invariant.

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
                        if from == to {
                            return Err(GraphError::SelfLoop);
                        }
                        self.edges.push((from, to));
                        self.degrees[from as usize] += 1;
                        self.degrees[to as usize] += 1;
                        Ok(())
                    }

                    pub fn remove_edge(&mut self, from: u32, to: u32) -> Result<(), GraphError> {
                        if let Some(position) = self.edges.iter().position(|edge| *edge == (from, to)) {
                            self.edges.remove(position);
                            self.degrees[from as usize] -= 1;
                            self.degrees[to as usize] -= 1;
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
            """),
        },
        "operation_items": [
            {"test": "add_updates_both_degrees",
             "operation": "apply the transition to every component carrying the invariant"},
            {"test": "remove_updates_both_degrees",
             "operation": "apply the inverse transition to both components symmetrically"},
            {"test": "a_self_loop_is_rejected_without_a_partial_update",
             "operation": "reject a transition that would violate the declared relation"},
            {"test": "a_missing_edge_is_rejected_without_a_partial_update",
             "operation": "precondition failure leaves the invariant untouched"},
            {"test": "conservation_holds_after_a_mixed_sequence",
             "operation": "the invariant is closed under mixed transitions"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "self.degrees[from as usize] -= 1;",
                     "replace": "self.degrees[from as usize] += 1;"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-06-invariant-substitution",
        "kind": "trio",
        "category": "trio",
        "difficulty": "hard",
        "capability": "Perform capture-avoiding substitution, rejecting a capturing transition.",
        "prompt_intro": (
            "The crate builds, but `substitute` breaks the free-variable invariant: it "
            "substitutes inside a lambda that shadows the name, and it captures a "
            "replacement's free variable. Make it capture-avoiding: a shadowed binder "
            "makes the substitution a no-op, and a substitution that would capture a "
            "replacement variable returns `None` with the term unchanged. Keep the "
            "signature."),
        "observed_failure": (
            "thread 'a_shadowed_binder_is_unchanged' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: Some(Lam(\"x\", Var(\"y\")))\n"
            " right: Some(Lam(\"x\", Var(\"x\")))"),
        "shared_operation": dict(_INVARIANT, surface="logic"),
        "files": {
            "src/lib.rs": code("""
                //! Naive substitution that ignores binding structure.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Term {
                    Var(String),
                    App(String, Vec<Term>),
                    Lam(String, Box<Term>),
                }

                pub fn free_vars(term: &Term) -> Vec<String> {
                    fn collect(term: &Term, bound: &mut Vec<String>, out: &mut Vec<String>) {
                        match term {
                            Term::Var(name) => {
                                if !bound.contains(name) {
                                    out.push(name.clone());
                                }
                            }
                            Term::App(_function, arguments) => {
                                for argument in arguments {
                                    collect(argument, bound, out);
                                }
                            }
                            Term::Lam(binder, body) => {
                                bound.push(binder.clone());
                                collect(body, bound, out);
                                bound.pop();
                            }
                        }
                    }
                    let mut out = Vec::new();
                    collect(term, &mut Vec::new(), &mut out);
                    out.sort();
                    out.dedup();
                    out
                }

                pub fn substitute(term: &Term, name: &str, replacement: &Term) -> Option<Term> {
                    Some(match term {
                        Term::Var(variable) if variable == name => replacement.clone(),
                        Term::Var(variable) => Term::Var(variable.clone()),
                        Term::App(function, arguments) => Term::App(
                            function.clone(),
                            arguments
                                .iter()
                                .map(|argument| substitute(argument, name, replacement).unwrap())
                                .collect(),
                        ),
                        Term::Lam(binder, body) => Term::Lam(
                            binder.clone(),
                            Box::new(substitute(body, name, replacement).unwrap()),
                        ),
                    })
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use trio_06_invariant_substitution::{free_vars, substitute, Term};

                fn var(name: &str) -> Term {
                    Term::Var(name.to_string())
                }

                #[test]
                fn replaces_a_free_variable() {
                    let term = Term::App("f".to_string(), vec![var("x")]);
                    let expected = Term::App("f".to_string(), vec![var("y")]);
                    assert_eq!(substitute(&term, "x", &var("y")), Some(expected));
                }

                #[test]
                fn a_shadowed_binder_is_unchanged() {
                    let term = Term::Lam("x".to_string(), Box::new(var("x")));
                    assert_eq!(substitute(&term, "x", &var("y")), Some(term));
                }

                #[test]
                fn a_capturing_substitution_is_rejected() {
                    let term = Term::Lam("y".to_string(), Box::new(var("x")));
                    assert_eq!(substitute(&term, "x", &var("y")), None);
                }

                #[test]
                fn substitution_descends_under_an_unrelated_binder() {
                    let term = Term::Lam("z".to_string(), Box::new(var("x")));
                    let expected = Term::Lam("z".to_string(), Box::new(var("y")));
                    assert_eq!(substitute(&term, "x", &var("y")), Some(expected));
                }

                #[test]
                fn the_free_variable_relation_is_preserved() {
                    let term = Term::Lam("z".to_string(), Box::new(var("x")));
                    let result = substitute(&term, "x", &var("y")).unwrap();
                    assert_eq!(free_vars(&result), vec!["y".to_string()]);
                    assert_eq!(free_vars(&term), vec!["x".to_string()]);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Capture-avoiding substitution.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Term {
                    Var(String),
                    App(String, Vec<Term>),
                    Lam(String, Box<Term>),
                }

                pub fn free_vars(term: &Term) -> Vec<String> {
                    fn collect(term: &Term, bound: &mut Vec<String>, out: &mut Vec<String>) {
                        match term {
                            Term::Var(name) => {
                                if !bound.contains(name) {
                                    out.push(name.clone());
                                }
                            }
                            Term::App(_function, arguments) => {
                                for argument in arguments {
                                    collect(argument, bound, out);
                                }
                            }
                            Term::Lam(binder, body) => {
                                bound.push(binder.clone());
                                collect(body, bound, out);
                                bound.pop();
                            }
                        }
                    }
                    let mut out = Vec::new();
                    collect(term, &mut Vec::new(), &mut out);
                    out.sort();
                    out.dedup();
                    out
                }

                pub fn substitute(term: &Term, name: &str, replacement: &Term) -> Option<Term> {
                    Some(match term {
                        Term::Var(variable) if variable == name => replacement.clone(),
                        Term::Var(variable) => Term::Var(variable.clone()),
                        Term::App(function, arguments) => Term::App(
                            function.clone(),
                            arguments
                                .iter()
                                .map(|argument| substitute(argument, name, replacement))
                                .collect::<Option<Vec<_>>>()?,
                        ),
                        Term::Lam(binder, body) => {
                            if binder == name {
                                return Some(term.clone());
                            }
                            if free_vars(replacement).iter().any(|variable| variable == binder) {
                                return None;
                            }
                            Term::Lam(
                                binder.clone(),
                                Box::new(substitute(body, name, replacement)?),
                            )
                        }
                    })
                }
            """),
        },
        "operation_items": [
            {"test": "replaces_a_free_variable",
             "operation": "apply the transition only to free occurrences"},
            {"test": "a_shadowed_binder_is_unchanged",
             "operation": "an inner binding masks the transition"},
            {"test": "a_capturing_substitution_is_rejected",
             "operation": "reject the transition when it would change the binding relation"},
            {"test": "substitution_descends_under_an_unrelated_binder",
             "operation": "apply the transition where no binding shadows it"},
            {"test": "the_free_variable_relation_is_preserved",
             "operation": "the declared free-variable invariant holds after the transition"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "return None;",
                     "replace": "return Some(term.clone());"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-07-scope-resolver",
        "kind": "trio",
        "category": "trio",
        "capability": "Resolve a name to the nearest enclosing binding in a scope chain.",
        "prompt_intro": (
            "The crate builds, but `resolve` returns the outermost matching binding "
            "instead of the nearest one. Return the distance to the nearest binding "
            "for the name (0 is the queried scope), or `None` when the name is free. "
            "Keep the public API."),
        "observed_failure": (
            "thread 'the_innermost_binding_wins' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: Some(1)\n"
            " right: Some(0)"),
        "shared_operation": dict(_QUANTIFIER, surface="rust"),
        "files": {
            "src/lib.rs": code("""
                //! Scope resolution, currently returning the outermost binding.

                #[derive(Clone, Debug, PartialEq)]
                pub struct Scope {
                    pub bindings: Vec<String>,
                    pub parent: Option<Box<Scope>>,
                }

                impl Scope {
                    pub fn root() -> Self {
                        Scope { bindings: Vec::new(), parent: None }
                    }

                    pub fn push(&self, binding: &str) -> Self {
                        Scope {
                            bindings: vec![binding.to_string()],
                            parent: Some(Box::new(self.clone())),
                        }
                    }
                }

                pub fn resolve(scope: &Scope, name: &str) -> Option<usize> {
                    let mut current = Some(scope);
                    let mut depth = 0;
                    let mut found = None;
                    while let Some(node) = current {
                        if node.bindings.iter().any(|binding| binding == name) {
                            found = Some(depth);
                        }
                        depth += 1;
                        current = node.parent.as_deref();
                    }
                    found
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use trio_07_scope_resolver::{resolve, Scope};

                #[test]
                fn the_innermost_binding_wins() {
                    let outer = Scope::root().push("x");
                    let inner = outer.push("x");
                    assert_eq!(resolve(&inner, "x"), Some(0));
                }

                #[test]
                fn an_outer_binding_is_found_at_its_distance() {
                    let outer = Scope::root().push("x");
                    let inner = outer.push("y");
                    assert_eq!(resolve(&inner, "x"), Some(1));
                }

                #[test]
                fn a_free_name_is_none() {
                    let scope = Scope::root().push("y");
                    assert_eq!(resolve(&scope, "x"), None);
                }

                #[test]
                fn shadowing_across_three_levels_resolves_innermost() {
                    let outer = Scope::root().push("x");
                    let middle = outer.push("y");
                    let inner = middle.push("x");
                    assert_eq!(resolve(&inner, "x"), Some(0));
                    assert_eq!(resolve(&inner, "y"), Some(1));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Nearest-binding scope resolution.

                #[derive(Clone, Debug, PartialEq)]
                pub struct Scope {
                    pub bindings: Vec<String>,
                    pub parent: Option<Box<Scope>>,
                }

                impl Scope {
                    pub fn root() -> Self {
                        Scope { bindings: Vec::new(), parent: None }
                    }

                    pub fn push(&self, binding: &str) -> Self {
                        Scope {
                            bindings: vec![binding.to_string()],
                            parent: Some(Box::new(self.clone())),
                        }
                    }
                }

                pub fn resolve(scope: &Scope, name: &str) -> Option<usize> {
                    let mut current = Some(scope);
                    let mut depth = 0;
                    while let Some(node) = current {
                        if node.bindings.iter().any(|binding| binding == name) {
                            return Some(depth);
                        }
                        depth += 1;
                        current = node.parent.as_deref();
                    }
                    None
                }
            """),
        },
        "operation_items": [
            {"test": "the_innermost_binding_wins",
             "operation": "resolve to the nearest enclosing binding"},
            {"test": "an_outer_binding_is_found_at_its_distance",
             "operation": "count the scopes crossed to the binding"},
            {"test": "a_free_name_is_none",
             "operation": "a name with no binding is free"},
            {"test": "shadowing_across_three_levels_resolves_innermost",
             "operation": "an inner scope never leaks past a shadowing binding"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "return Some(depth);",
                     "replace": "return Some(depth + 1);"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-08-scoped-quantifier-eval",
        "kind": "trio",
        "category": "trio",
        "difficulty": "hard",
        "capability": "Evaluate scoped sum binders so an inner binder shadows an outer one.",
        "prompt_intro": (
            "The crate builds, but `eval` looks a variable up from the outermost "
            "environment binding, so an inner `Sum` binder does not shadow an outer "
            "one. Resolve variables against the nearest binding, keep an unbound "
            "variable an error `unbound <name>`, and evaluate `Sum` over the closed "
            "integer range. Keep the public API."),
        "observed_failure": (
            "thread 'an_inner_binder_shadows_an_outer_one' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: 202\n"
            " right: 5"),
        "shared_operation": dict(_QUANTIFIER, surface="numeric-graph"),
        "files": {
            "src/lib.rs": code("""
                //! Scoped arithmetic evaluation, currently resolving outermost.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Expr {
                    Const(i64),
                    Var(String),
                    Add(Box<Expr>, Box<Expr>),
                    Mul(Box<Expr>, Box<Expr>),
                    Sum { variable: String, low: i64, high: i64, body: Box<Expr> },
                }

                pub fn eval(expr: &Expr, environment: &[(String, i64)]) -> Result<i64, String> {
                    match expr {
                        Expr::Const(value) => Ok(*value),
                        Expr::Var(name) => environment
                            .iter()
                            .find(|(binding, _value)| binding == name)
                            .map(|(_binding, value)| *value)
                            .ok_or_else(|| format!("unbound {name}")),
                        Expr::Add(left, right) => Ok(eval(left, environment)? + eval(right, environment)?),
                        Expr::Mul(left, right) => Ok(eval(left, environment)? * eval(right, environment)?),
                        Expr::Sum { variable, low, high, body } => {
                            let mut total = 0;
                            let mut inner = environment.to_vec();
                            for value in *low..=*high {
                                inner.push((variable.clone(), value));
                                total += eval(body, &inner)?;
                            }
                            Ok(total)
                        }
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use trio_08_scoped_quantifier_eval::{eval, Expr};

                fn var(name: &str) -> Expr {
                    Expr::Var(name.to_string())
                }

                fn add(left: Expr, right: Expr) -> Expr {
                    Expr::Add(Box::new(left), Box::new(right))
                }

                fn mul(left: Expr, right: Expr) -> Expr {
                    Expr::Mul(Box::new(left), Box::new(right))
                }

                fn sum(variable: &str, low: i64, high: i64, body: Expr) -> Expr {
                    Expr::Sum {
                        variable: variable.to_string(),
                        low,
                        high,
                        body: Box::new(body),
                    }
                }

                #[test]
                fn a_sum_evaluates_its_body() {
                    let expr = sum("x", 1, 4, mul(var("x"), var("x")));
                    assert_eq!(eval(&expr, &[]), Ok(30));
                }

                #[test]
                fn an_inner_binder_shadows_an_outer_one() {
                    let outer = vec![("x".to_string(), 100)];
                    let expr = sum("x", 1, 2, add(var("x"), Expr::Const(1)));
                    assert_eq!(eval(&expr, &outer), Ok(5));
                }

                #[test]
                fn an_unbound_variable_is_an_error() {
                    assert_eq!(eval(&var("z"), &[]), Err("unbound z".to_string()));
                }

                #[test]
                fn an_outer_binding_remains_visible_under_an_unrelated_binder() {
                    let outer = vec![("y".to_string(), 7)];
                    let expr = sum("x", 1, 2, var("y"));
                    assert_eq!(eval(&expr, &outer), Ok(14));
                }

                #[test]
                fn arithmetic_nesting_is_respected() {
                    let expr = sum("x", 1, 3, add(mul(Expr::Const(2), var("x")), Expr::Const(1)));
                    assert_eq!(eval(&expr, &[]), Ok(15));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Scoped arithmetic evaluation with nearest-binding resolution.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Expr {
                    Const(i64),
                    Var(String),
                    Add(Box<Expr>, Box<Expr>),
                    Mul(Box<Expr>, Box<Expr>),
                    Sum { variable: String, low: i64, high: i64, body: Box<Expr> },
                }

                pub fn eval(expr: &Expr, environment: &[(String, i64)]) -> Result<i64, String> {
                    match expr {
                        Expr::Const(value) => Ok(*value),
                        Expr::Var(name) => environment
                            .iter()
                            .rev()
                            .find(|(binding, _value)| binding == name)
                            .map(|(_binding, value)| *value)
                            .ok_or_else(|| format!("unbound {name}")),
                        Expr::Add(left, right) => Ok(eval(left, environment)? + eval(right, environment)?),
                        Expr::Mul(left, right) => Ok(eval(left, environment)? * eval(right, environment)?),
                        Expr::Sum { variable, low, high, body } => {
                            let mut total = 0;
                            let mut inner = environment.to_vec();
                            for value in *low..=*high {
                                inner.push((variable.clone(), value));
                                total += eval(body, &inner)?;
                            }
                            Ok(total)
                        }
                    }
                }
            """),
        },
        "operation_items": [
            {"test": "a_sum_evaluates_its_body",
             "operation": "the binder governs occurrences in its body"},
            {"test": "an_inner_binder_shadows_an_outer_one",
             "operation": "resolve to the nearest binding, not the outermost"},
            {"test": "an_unbound_variable_is_an_error",
             "operation": "a name outside every scope is free"},
            {"test": "an_outer_binding_remains_visible_under_an_unrelated_binder",
             "operation": "a non-matching binder does not change the scope of another name"},
            {"test": "arithmetic_nesting_is_respected",
             "operation": "scope resolution composes with the expression structure"},
        ],
        "mutation": {"file": "src/lib.rs", "find": ".iter()\n            .rev()",
                     "replace": ".iter()"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-09-quantifier-occurrences",
        "kind": "trio",
        "category": "trio",
        "difficulty": "hard",
        "capability": "Classify variable occurrences as free or bound by the nearest matching quantifier.",
        "prompt_intro": (
            "The crate builds, but `occurrences` increments the binding depth for "
            "every quantifier instead of only for a quantifier binding the queried "
            "name, so a variable under an unrelated binder is misclassified. "
            "Classify each occurrence as `Free` or `Bound(d)`, where `d` is one "
            "less than the number of same-name binders enclosing the occurrence "
            "(zero for the outermost). Keep the public API."),
        "observed_failure": (
            "thread 'a_variable_under_an_unrelated_binder_is_free' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: [Bound(0)]\n"
            " right: [Free]"),
        "shared_operation": dict(_QUANTIFIER, surface="logic"),
        "files": {
            "src/lib.rs": code("""
                //! Occurrence classification, currently counting every binder.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Occurrence {
                    Free,
                    Bound(u32),
                }

                #[derive(Clone, Debug, PartialEq)]
                pub enum Formula {
                    Var(String),
                    Atom(String),
                    Not(Box<Formula>),
                    And(Box<Formula>, Box<Formula>),
                    Forall(String, Box<Formula>),
                    Exists(String, Box<Formula>),
                }

                pub fn occurrences(formula: &Formula, name: &str) -> Vec<Occurrence> {
                    fn walk(formula: &Formula, name: &str, depth: u32, out: &mut Vec<Occurrence>) {
                        match formula {
                            Formula::Var(variable) => {
                                if variable == name {
                                    out.push(if depth == 0 {
                                        Occurrence::Free
                                    } else {
                                        Occurrence::Bound(depth - 1)
                                    });
                                }
                            }
                            Formula::Atom(_) => {}
                            Formula::Not(inner) => walk(inner, name, depth, out),
                            Formula::And(left, right) => {
                                walk(left, name, depth, out);
                                walk(right, name, depth, out);
                            }
                            Formula::Forall(_binder, body) => walk(body, name, depth + 1, out),
                            Formula::Exists(_binder, body) => walk(body, name, depth + 1, out),
                        }
                    }
                    let mut out = Vec::new();
                    walk(formula, name, 0, &mut out);
                    out
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use trio_09_quantifier_occurrences::{occurrences, Formula, Occurrence};

                fn var(name: &str) -> Formula {
                    Formula::Var(name.to_string())
                }

                #[test]
                fn a_bound_occurrence_is_bound_at_depth_zero() {
                    let formula = Formula::Forall("x".to_string(), Box::new(var("x")));
                    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(0)]);
                }

                #[test]
                fn free_and_bound_occurrences_are_distinguished() {
                    let formula = Formula::And(
                        Box::new(var("x")),
                        Box::new(Formula::Forall("x".to_string(), Box::new(var("x")))),
                    );
                    assert_eq!(
                        occurrences(&formula, "x"),
                        vec![Occurrence::Free, Occurrence::Bound(0)]
                    );
                }

                #[test]
                fn a_variable_under_an_unrelated_binder_is_free() {
                    let formula = Formula::Exists("y".to_string(), Box::new(var("x")));
                    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Free]);
                }

                #[test]
                fn an_outer_variable_is_bound_at_the_nearest_same_name_binder() {
                    let formula = Formula::Forall(
                        "x".to_string(),
                        Box::new(Formula::Exists("y".to_string(), Box::new(var("x")))),
                    );
                    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(0)]);
                }

                #[test]
                fn nested_same_name_binders_increase_the_binding_depth() {
                    let formula = Formula::Forall(
                        "x".to_string(),
                        Box::new(Formula::Forall("x".to_string(), Box::new(var("x")))),
                    );
                    assert_eq!(occurrences(&formula, "x"), vec![Occurrence::Bound(1)]);
                }

                #[test]
                fn a_formula_without_the_name_has_no_occurrences() {
                    let formula = Formula::And(
                        Box::new(Formula::Atom("p".to_string())),
                        Box::new(Formula::Atom("q".to_string())),
                    );
                    assert!(occurrences(&formula, "x").is_empty());
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Occurrence classification by the nearest matching binder.

                #[derive(Clone, Debug, PartialEq)]
                pub enum Occurrence {
                    Free,
                    Bound(u32),
                }

                #[derive(Clone, Debug, PartialEq)]
                pub enum Formula {
                    Var(String),
                    Atom(String),
                    Not(Box<Formula>),
                    And(Box<Formula>, Box<Formula>),
                    Forall(String, Box<Formula>),
                    Exists(String, Box<Formula>),
                }

                pub fn occurrences(formula: &Formula, name: &str) -> Vec<Occurrence> {
                    fn walk(formula: &Formula, name: &str, depth: u32, out: &mut Vec<Occurrence>) {
                        match formula {
                            Formula::Var(variable) => {
                                if variable == name {
                                    out.push(if depth == 0 {
                                        Occurrence::Free
                                    } else {
                                        Occurrence::Bound(depth - 1)
                                    });
                                }
                            }
                            Formula::Atom(_) => {}
                            Formula::Not(inner) => walk(inner, name, depth, out),
                            Formula::And(left, right) => {
                                walk(left, name, depth, out);
                                walk(right, name, depth, out);
                            }
                            Formula::Forall(binder, body) => {
                                let next = if binder == name { depth + 1 } else { depth };
                                walk(body, name, next, out);
                            }
                            Formula::Exists(binder, body) => {
                                let next = if binder == name { depth + 1 } else { depth };
                                walk(body, name, next, out);
                            }
                        }
                    }
                    let mut out = Vec::new();
                    walk(formula, name, 0, &mut out);
                    out
                }
            """),
        },
        "operation_items": [
            {"test": "a_bound_occurrence_is_bound_at_depth_zero",
             "operation": "a matching binder governs occurrences in its body"},
            {"test": "free_and_bound_occurrences_are_distinguished",
             "operation": "resolve each occurrence against its own scope"},
            {"test": "a_variable_under_an_unrelated_binder_is_free",
             "operation": "a non-matching binder does not bind the name"},
            {"test": "an_outer_variable_is_bound_at_the_nearest_same_name_binder",
             "operation": "skip unrelated binders when counting depth"},
            {"test": "nested_same_name_binders_increase_the_binding_depth",
             "operation": "each enclosing same-name binder increases the binding depth"},
            {"test": "a_formula_without_the_name_has_no_occurrences",
             "operation": "the empty occurrence set is stable"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "if binder == name { depth + 1 } else { depth }",
                     "replace": "if binder != name { depth + 1 } else { depth }"},
        "initial_failure": {"kind": "hidden_test_failure",
                            "must_contain": ["test result: FAILED"]},
    },
]
