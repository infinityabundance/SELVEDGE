use failing_09_state_cycle::{next, Light};

#[test]
fn red_to_green() {
    assert_eq!(next(Light::Red), Light::Green);
}

#[test]
fn green_to_yellow() {
    assert_eq!(next(Light::Green), Light::Yellow);
}

#[test]
fn yellow_to_red() {
    assert_eq!(next(Light::Yellow), Light::Red);
}

#[test]
fn three_steps_return_home() {
    let start = Light::Red;
    assert_eq!(next(next(next(start))), start);
}
