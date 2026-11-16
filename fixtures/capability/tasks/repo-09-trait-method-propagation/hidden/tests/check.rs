use repo_09_trait_method_propagation::{csv::Csv, json::Json, pipeline::Sink, run};

#[test]
fn pipeline_joins_every_sink() {
    assert_eq!(run(3), "{\"v\":3};3;ok");
}

#[test]
fn pipeline_handles_zero() {
    assert_eq!(run(0), "{\"v\":0};0;ok");
}

#[test]
fn pipeline_handles_negatives() {
    assert_eq!(run(-1), "{\"v\":-1};-1;ok");
}

#[test]
fn every_sink_flushes() {
    assert_eq!(Json.flush(), "ok");
    assert_eq!(Csv.flush(), "ok");
}
