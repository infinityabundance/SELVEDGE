"""The anti-superficial-specialisation trio (P11.21).

Three surfaces, one intended operation. Each task is a small crate whose hidden
tests cannot be passed by matching the surface form, because the same
dependency-closure repair is required in each:

* **rust-lifetime-chain** — annotate a lifetime through a two-call chain so the
  returned borrow is tied to the table and not to a short-lived alias slice;
* **math-dependency** — return the minimal dependency-ordered recompute set for
  a target, excluding known quantities and rejecting cycles;
* **logic-quantifier** — return the shortest premise chain that establishes a
  quantified conclusion, and list the premises whose quantifier must be
  strengthened from `Some` to `All`.

The shared operation, annotated per hidden test below, is: *build the dependency
relation induced by a target, close it from the required node, reject an
inconsistent cycle, and emit the minimal ordered set of declarations that must
change*. A win is judged on the trio as a unit: the same arm must succeed on all
three surfaces, the per-surface breakdown is reported, and a pooled gain driven
by one surface is not evidence of transfer.

Honest boundary: this is three tasks with five operation-annotated tests each,
annotated by the author. P11.21 asks for >= 40 items per domain and an
inter-annotator agreement check; neither exists yet, and `docs/CAPABILITY-COURT.md`
records that as an open gap rather than a claim.
"""

_OPERATION = {
    "name": "dependency-closure minimal repair",
    "definition": (
        "Given a target, build its dependency relation, close it from the required "
        "node while excluding already-satisfied nodes, reject a cycle as an error, "
        "and emit the minimal set of declarations to change in dependency order."),
    "transfer_rule": (
        "A trio win requires success on every surface with the per-surface "
        "breakdown reported; a pooled or single-surface gain is not transfer."),
    "audit": "author annotation only; no inter-annotator agreement yet (P11.21 gap)",
}

TASKS = [
    {
        "id": "trio-01-lifetime-chain",
        "kind": "trio",
        "category": "trio",
        "capability": "Propagate a lifetime constraint across a lookup/resolve/chain call graph.",
        "prompt_intro": (
            "This crate does not build. `lookup`, `resolve` and `chain` each elide a\n"
            "lifetime across two or more input references. Annotate the lifetime so every\n"
            "returned `&str` is tied to the table it came from, never to the alias slice.\n"
            "Semantics: `resolve` looks the key up directly, then through the alias table;\n"
            "`chain` delegates to `resolve`."),
        "observed_failure": (
            "error[E0106]: missing lifetime specifier\n"
            " --> src/lib.rs:5:52\n"
            "  |\n"
            "5 | pub fn lookup(table: &Table, key: &str) -> Option<&str> {\n"
            "  |                     -----         ----            ^ expected named lifetime parameter"),
        "shared_operation": dict(_OPERATION, surface="rust"),
        "files": {
            "src/lib.rs": (
                "//! A table, an alias table, and a resolution chain.\n"
                "\n"
                "pub type Table = [(&'static str, &'static str)];\n"
                "\n"
                "pub fn lookup(table: &Table, key: &str) -> Option<&str> {\n"
                "    table\n"
                "        .iter()\n"
                "        .find(|(name, _value)| *name == key)\n"
                "        .map(|(_name, value)| *value)\n"
                "}\n"
                "\n"
                "pub fn resolve(\n"
                "    table: &Table,\n"
                "    aliases: &[(&'static str, &'static str)],\n"
                "    key: &str,\n"
                ") -> Option<&str> {\n"
                "    if let Some(value) = lookup(table, key) {\n"
                "        return Some(value);\n"
                "    }\n"
                "    aliases\n"
                "        .iter()\n"
                "        .find(|(alias, _target)| *alias == key)\n"
                "        .and_then(|(_alias, target)| lookup(table, target))\n"
                "}\n"
                "\n"
                "pub fn chain(\n"
                "    table: &Table,\n"
                "    aliases: &[(&'static str, &'static str)],\n"
                "    key: &str,\n"
                ") -> Option<&str> {\n"
                "    resolve(table, aliases, key)\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use trio_01_lifetime_chain::{chain, lookup, resolve};\n"
                "\n"
                "fn table() -> Vec<(&'static str, &'static str)> {\n"
                "    vec![(\"alpha\", \"one\"), (\"beta\", \"two\")]\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn direct_lookup_borrows_the_table() {\n"
                "    let table = table();\n"
                "    assert_eq!(lookup(&table, \"alpha\"), Some(\"one\"));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn resolution_follows_one_alias() {\n"
                "    let table = table();\n"
                "    let aliases = [(\"a\", \"alpha\")];\n"
                "    assert_eq!(resolve(&table, &aliases, \"a\"), Some(\"one\"));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_chain_result_outlives_the_alias_slice() {\n"
                "    let table = table();\n"
                "    let value = {\n"
                "        let aliases = [(\"a\", \"alpha\")];\n"
                "        chain(&table, &aliases, \"a\")\n"
                "    };\n"
                "    assert_eq!(value, Some(\"one\"));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn an_unknown_key_is_none() {\n"
                "    let table = table();\n"
                "    let aliases = [(\"a\", \"alpha\")];\n"
                "    assert_eq!(chain(&table, &aliases, \"missing\"), None);\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_direct_hit_skips_the_alias_table() {\n"
                "    let table = table();\n"
                "    let aliases = [(\"alpha\", \"beta\")];\n"
                "    assert_eq!(chain(&table, &aliases, \"alpha\"), Some(\"one\"));\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! A table, an alias table, and a resolution chain.\n"
                "\n"
                "pub type Table = [(&'static str, &'static str)];\n"
                "\n"
                "pub fn lookup<'a>(table: &'a Table, key: &str) -> Option<&'a str> {\n"
                "    table\n"
                "        .iter()\n"
                "        .find(|(name, _value)| *name == key)\n"
                "        .map(|(_name, value)| *value)\n"
                "}\n"
                "\n"
                "pub fn resolve<'a>(\n"
                "    table: &'a Table,\n"
                "    aliases: &[(&'static str, &'static str)],\n"
                "    key: &str,\n"
                ") -> Option<&'a str> {\n"
                "    if let Some(value) = lookup(table, key) {\n"
                "        return Some(value);\n"
                "    }\n"
                "    aliases\n"
                "        .iter()\n"
                "        .find(|(alias, _target)| *alias == key)\n"
                "        .and_then(|(_alias, target)| lookup(table, target))\n"
                "}\n"
                "\n"
                "pub fn chain<'a>(\n"
                "    table: &'a Table,\n"
                "    aliases: &[(&'static str, &'static str)],\n"
                "    key: &str,\n"
                ") -> Option<&'a str> {\n"
                "    resolve(table, aliases, key)\n"
                "}\n"),
        },
        "operation_items": [
            {"test": "direct_lookup_borrows_the_table",
             "operation": "close the target `value` over the table dependency"},
            {"test": "resolution_follows_one_alias",
             "operation": "close over alias -> table in dependency order"},
            {"test": "the_chain_result_outlives_the_alias_slice",
             "operation": "reject the inconsistent (shorter) provenance and keep the "
                          "minimal consistent repair"},
            {"test": "an_unknown_key_is_none",
             "operation": "a target outside the relation is not repaired"},
            {"test": "a_direct_hit_skips_the_alias_table",
             "operation": "exclude already-satisfied declarations from the repair set"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "*name == key",
                     "replace": "*name != key"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0106"]},
    },
    {
        "id": "trio-02-math-dependency",
        "kind": "trio",
        "category": "trio",
        "capability": "Compute a minimal topological recompute set, excluding known quantities and rejecting cycles.",
        "prompt_intro": (
            "`minimal_recompute` must return the outputs that have to be recomputed to\n"
            "obtain `target`, in dependency order, with each output appearing once and\n"
            "with outputs in `known` omitted (and their dependencies pruned). If `target`\n"
            "is already known, return an empty plan. If the target has no rule and is not\n"
            "known, return `Err(PlanError::Unknown(target))`. If the dependency closure\n"
            "contains a cycle, return `Err(PlanError::Cycle(node))` with the node that\n"
            "closes the cycle. Ties are broken by the order rules are declared."),
        "observed_failure": (
            "thread 'tests::known_inputs_are_pruned' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: [\"c\", \"b\"]\n"
            " right: [\"b\", \"c\"]"),
        "shared_operation": dict(_OPERATION, surface="numeric-graph"),
        "files": {
            "src/lib.rs": (
                "//! Recompute planning over declared rules.\n"
                "\n"
                "#[derive(Debug, Clone, Copy)]\n"
                "pub struct Rule {\n"
                "    pub output: &'static str,\n"
                "    pub inputs: &'static [&'static str],\n"
                "}\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum PlanError {\n"
                "    Unknown(&'static str),\n"
                "    Cycle(&'static str),\n"
                "}\n"
                "\n"
                "pub fn minimal_recompute(\n"
                "    rules: &[Rule],\n"
                "    known: &[&str],\n"
                "    target: &'static str,\n"
                ") -> Result<Vec<&'static str>, PlanError> {\n"
                "    if known.contains(&target) {\n"
                "        return Ok(Vec::new());\n"
                "    }\n"
                "    let mut plan = Vec::new();\n"
                "    let mut stack = vec![target];\n"
                "    let mut visited: Vec<&'static str> = Vec::new();\n"
                "    while let Some(node) = stack.pop() {\n"
                "        if visited.contains(&node) {\n"
                "            continue;\n"
                "        }\n"
                "        visited.push(node);\n"
                "        let rule = rules\n"
                "            .iter()\n"
                "            .find(|rule| rule.output == node)\n"
                "            .ok_or(PlanError::Unknown(node))?;\n"
                "        plan.push(node);\n"
                "        for input in rule.inputs {\n"
                "            if !known.contains(input) {\n"
                "                stack.push(input);\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    Ok(plan)\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use trio_02_math_dependency::{minimal_recompute, PlanError, Rule};\n"
                "\n"
                "const RULES: &[Rule] = &[\n"
                "    Rule { output: \"b\", inputs: &[\"a\"] },\n"
                "    Rule { output: \"c\", inputs: &[\"b\"] },\n"
                "];\n"
                "\n"
                "fn plan<'a>(\n"
                "    rules: &'a [Rule],\n"
                "    known: &[&str],\n"
                "    target: &'static str,\n"
                ") -> Result<Vec<&'static str>, PlanError> {\n"
                "    minimal_recompute(rules, known, target)\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn known_inputs_are_pruned() {\n"
                "    assert_eq!(plan(RULES, &[\"a\"], \"c\"), Ok(vec![\"b\", \"c\"]));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_known_target_needs_nothing() {\n"
                "    assert_eq!(plan(RULES, &[\"a\", \"b\", \"c\"], \"c\"), Ok(vec![]));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn intermediate_known_values_are_skipped() {\n"
                "    assert_eq!(plan(RULES, &[\"a\", \"b\"], \"c\"), Ok(vec![\"c\"]));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_diamond_orders_dependencies_first() {\n"
                "    let rules = [\n"
                "        Rule { output: \"b\", inputs: &[\"a\"] },\n"
                "        Rule { output: \"c\", inputs: &[\"a\"] },\n"
                "        Rule { output: \"d\", inputs: &[\"b\", \"c\"] },\n"
                "    ];\n"
                "    assert_eq!(plan(&rules, &[\"a\"], \"d\"), Ok(vec![\"b\", \"c\", \"d\"]));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn an_unknown_target_is_an_error() {\n"
                "    assert_eq!(plan(RULES, &[\"a\"], \"zz\"), Err(PlanError::Unknown(\"zz\")));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_cycle_is_an_error() {\n"
                "    let rules = [\n"
                "        Rule { output: \"a\", inputs: &[\"b\"] },\n"
                "        Rule { output: \"b\", inputs: &[\"a\"] },\n"
                "    ];\n"
                "    assert_eq!(plan(&rules, &[], \"a\"), Err(PlanError::Cycle(\"a\")));\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Recompute planning over declared rules.\n"
                "\n"
                "#[derive(Debug, Clone, Copy)]\n"
                "pub struct Rule {\n"
                "    pub output: &'static str,\n"
                "    pub inputs: &'static [&'static str],\n"
                "}\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub enum PlanError {\n"
                "    Unknown(&'static str),\n"
                "    Cycle(&'static str),\n"
                "}\n"
                "\n"
                "fn visit(\n"
                "    rules: &[Rule],\n"
                "    known: &[&str],\n"
                "    node: &'static str,\n"
                "    done: &mut Vec<&'static str>,\n"
                "    visiting: &mut Vec<&'static str>,\n"
                "    plan: &mut Vec<&'static str>,\n"
                ") -> Result<(), PlanError> {\n"
                "    if known.contains(&node) || done.contains(&node) {\n"
                "        return Ok(());\n"
                "    }\n"
                "    if visiting.contains(&node) {\n"
                "        return Err(PlanError::Cycle(node));\n"
                "    }\n"
                "    let rule = rules\n"
                "        .iter()\n"
                "        .find(|rule| rule.output == node)\n"
                "        .ok_or(PlanError::Unknown(node))?;\n"
                "    visiting.push(node);\n"
                "    for input in rule.inputs {\n"
                "        visit(rules, known, input, done, visiting, plan)?;\n"
                "    }\n"
                "    visiting.pop();\n"
                "    done.push(node);\n"
                "    plan.push(node);\n"
                "    Ok(())\n"
                "}\n"
                "\n"
                "pub fn minimal_recompute(\n"
                "    rules: &[Rule],\n"
                "    known: &[&str],\n"
                "    target: &'static str,\n"
                ") -> Result<Vec<&'static str>, PlanError> {\n"
                "    if known.contains(&target) {\n"
                "        return Ok(Vec::new());\n"
                "    }\n"
                "    let mut plan = Vec::new();\n"
                "    let mut done = Vec::new();\n"
                "    let mut visiting = Vec::new();\n"
                "    visit(rules, known, target, &mut done, &mut visiting, &mut plan)?;\n"
                "    Ok(plan)\n"
                "}\n"),
        },
        "operation_items": [
            {"test": "known_inputs_are_pruned",
             "operation": "close from the target, exclude satisfied nodes, emit in order"},
            {"test": "a_known_target_needs_nothing",
             "operation": "an empty closure"},
            {"test": "intermediate_known_values_are_skipped",
             "operation": "prune at depth and keep the closure minimal"},
            {"test": "a_diamond_orders_dependencies_first",
             "operation": "topological order with deduplication"},
            {"test": "an_unknown_target_is_an_error",
             "operation": "a target outside the relation is rejected"},
            {"test": "a_cycle_is_an_error",
             "operation": "reject an inconsistent cycle and name its closing node"},
        ],
        "mutation": {"file": "src/lib.rs",
                     "find": "if known.contains(&node) || done.contains(&node) {",
                     "replace": "if done.contains(&node) {"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
    {
        "id": "trio-03-logic-quantifier",
        "kind": "trio",
        "category": "trio",
        "capability": "Find the shortest premise chain for a quantified conclusion and the minimal strengthening set.",
        "prompt_intro": (
            "`repair_path` must find the shortest premise chain from `from` to `to` and\n"
            "report which premises in that chain must be strengthened from `Some` to\n"
            "`All`. `path` holds the premise indices in order; `strengthened` holds the\n"
            "indices in that path whose quantifier was `Some`. If `from == to`, return an\n"
            "empty path. If no chain exists, return `None`. Ties on length are broken by\n"
            "the lexicographically smallest index sequence."),
        "observed_failure": (
            "thread 'tests::a_two_step_chain_is_found' panicked at tests/check.rs:\n"
            "assertion `left == right` failed\n"
            "  left: None\n"
            " right: Some([0, 1])"),
        "shared_operation": dict(_OPERATION, surface="logic"),
        "files": {
            "src/lib.rs": (
                "//! Quantified premise chains.\n"
                "\n"
                "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n"
                "pub enum Quantifier {\n"
                "    All,\n"
                "    Some,\n"
                "}\n"
                "\n"
                "#[derive(Debug, Clone, Copy)]\n"
                "pub struct Premise {\n"
                "    pub subject: &'static str,\n"
                "    pub quantifier: Quantifier,\n"
                "    pub object: &'static str,\n"
                "}\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub struct Repair {\n"
                "    pub path: Vec<usize>,\n"
                "    pub strengthened: Vec<usize>,\n"
                "}\n"
                "\n"
                "pub fn repair_path(\n"
                "    premises: &[Premise],\n"
                "    from: &'static str,\n"
                "    to: &'static str,\n"
                ") -> Option<Repair> {\n"
                "    if from == to {\n"
                "        return Some(Repair { path: Vec::new(), strengthened: Vec::new() });\n"
                "    }\n"
                "    premises\n"
                "        .iter()\n"
                "        .position(|premise| premise.subject == from && premise.object == to)\n"
                "        .map(|index| Repair { path: vec![index], strengthened: Vec::new() })\n"
                "}\n"),
        },
        "hidden": {
            "tests/check.rs": (
                "use trio_03_logic_quantifier::{repair_path, Premise, Quantifier, Repair};\n"
                "\n"
                "const ALL: Quantifier = Quantifier::All;\n"
                "const SOME: Quantifier = Quantifier::Some;\n"
                "\n"
                "fn premise(subject: &'static str, quantifier: Quantifier, object: &'static str) -> Premise {\n"
                "    Premise { subject, quantifier, object }\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_direct_all_premise_needs_no_strengthening() {\n"
                "    let premises = [premise(\"a\", ALL, \"b\")];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"b\"),\n"
                "               Some(Repair { path: vec![0], strengthened: vec![] }));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_two_step_chain_is_found() {\n"
                "    let premises = [premise(\"a\", ALL, \"b\"), premise(\"b\", ALL, \"c\")];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"c\"),\n"
                "               Some(Repair { path: vec![0, 1], strengthened: vec![] }));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn a_some_link_must_be_strengthened() {\n"
                "    let premises = [premise(\"a\", SOME, \"b\"), premise(\"b\", ALL, \"c\")];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"c\"),\n"
                "               Some(Repair { path: vec![0, 1], strengthened: vec![0] }));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn the_shortest_chain_wins() {\n"
                "    let premises = [\n"
                "        premise(\"a\", ALL, \"x\"),\n"
                "        premise(\"x\", ALL, \"z\"),\n"
                "        premise(\"a\", ALL, \"z\"),\n"
                "    ];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"z\"),\n"
                "               Some(Repair { path: vec![2], strengthened: vec![] }));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn identical_nodes_need_no_premise() {\n"
                "    let premises: [Premise; 0] = [];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"a\"),\n"
                "               Some(Repair { path: vec![], strengthened: vec![] }));\n"
                "}\n"
                "\n"
                "#[test]\n"
                "fn an_unreachable_target_is_none() {\n"
                "    let premises = [premise(\"a\", ALL, \"b\")];\n"
                "    assert_eq!(repair_path(&premises, \"a\", \"zz\"), None);\n"
                "}\n"),
        },
        "reference": {
            "src/lib.rs": (
                "//! Quantified premise chains.\n"
                "\n"
                "use std::collections::VecDeque;\n"
                "\n"
                "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\n"
                "pub enum Quantifier {\n"
                "    All,\n"
                "    Some,\n"
                "}\n"
                "\n"
                "#[derive(Debug, Clone, Copy)]\n"
                "pub struct Premise {\n"
                "    pub subject: &'static str,\n"
                "    pub quantifier: Quantifier,\n"
                "    pub object: &'static str,\n"
                "}\n"
                "\n"
                "#[derive(Debug, PartialEq)]\n"
                "pub struct Repair {\n"
                "    pub path: Vec<usize>,\n"
                "    pub strengthened: Vec<usize>,\n"
                "}\n"
                "\n"
                "pub fn repair_path(\n"
                "    premises: &[Premise],\n"
                "    from: &'static str,\n"
                "    to: &'static str,\n"
                ") -> Option<Repair> {\n"
                "    if from == to {\n"
                "        return Some(Repair { path: Vec::new(), strengthened: Vec::new() });\n"
                "    }\n"
                "    let mut queue: VecDeque<(&'static str, Vec<usize>)> = VecDeque::new();\n"
                "    let mut seen: Vec<&'static str> = vec![from];\n"
                "    queue.push_back((from, Vec::new()));\n"
                "    while let Some((node, path)) = queue.pop_front() {\n"
                "        for (index, premise) in premises.iter().enumerate() {\n"
                "            if premise.subject != node || seen.contains(&premise.object) {\n"
                "                continue;\n"
                "            }\n"
                "            let mut next_path = path.clone();\n"
                "            next_path.push(index);\n"
                "            if premise.object == to {\n"
                "                let strengthened = next_path\n"
                "                    .iter()\n"
                "                    .copied()\n"
                "                    .filter(|index| premises[*index].quantifier == Quantifier::Some)\n"
                "                    .collect();\n"
                "                return Some(Repair { path: next_path, strengthened });\n"
                "            }\n"
                "            seen.push(premise.object);\n"
                "            queue.push_back((premise.object, next_path));\n"
                "        }\n"
                "    }\n"
                "    None\n"
                "}\n"),
        },
        "operation_items": [
            {"test": "a_direct_all_premise_needs_no_strengthening",
             "operation": "one-edge closure, no repair required"},
            {"test": "a_two_step_chain_is_found",
             "operation": "close the target over the premise relation and order it"},
            {"test": "a_some_link_must_be_strengthened",
             "operation": "reduce the edge to the required consistency rule"},
            {"test": "the_shortest_chain_wins",
             "operation": "choose the minimal closure"},
            {"test": "identical_nodes_need_no_premise",
             "operation": "an empty closure"},
            {"test": "an_unreachable_target_is_none",
             "operation": "a target outside the relation is rejected"},
        ],
        "mutation": {"file": "src/lib.rs", "find": "if premise.object == to {",
                     "replace": "if premise.object == to && path.is_empty() {"},
        "initial_failure": {"kind": "test_failure", "must_contain": ["test result: FAILED"]},
    },
]
