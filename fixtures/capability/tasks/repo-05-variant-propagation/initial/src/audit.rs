//! Audit descriptions.

use crate::event::Event;

pub fn describe(event: &Event) -> String {
    match event {
        Event::Start => "S".to_string(),
        Event::Stop => "T".to_string(),
    }
}
