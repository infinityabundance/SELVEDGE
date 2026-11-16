//! Event handling.

use crate::event::Event;

pub fn handle(event: &Event) -> &'static str {
    match event {
        Event::Start => "start",
        Event::Stop => "stop",
        Event::Pause => "pause",
    }
}
