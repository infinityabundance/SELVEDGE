use repo_05_variant_propagation::{audit::describe, event::Event, handle};

#[test]
fn handles_every_event() {
    assert_eq!(handle(&Event::Start), "start");
    assert_eq!(handle(&Event::Stop), "stop");
    assert_eq!(handle(&Event::Pause), "pause");
}

#[test]
fn describes_every_event() {
    assert_eq!(describe(&Event::Start), "S");
    assert_eq!(describe(&Event::Stop), "T");
    assert_eq!(describe(&Event::Pause), "P");
}

#[test]
fn descriptions_are_one_character() {
    for event in [Event::Start, Event::Stop, Event::Pause] {
        assert_eq!(describe(&event).chars().count(), 1);
    }
}
