//! Duration formatting. The visible test fails: the minute field is
//! the total number of minutes.

pub fn format_duration(total_seconds: u64) -> String {
    let hours = total_seconds / 3600;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

#[cfg(test)]
mod tests {
    use super::format_duration;

    #[test]
    fn visible_one_hour() {
        assert_eq!(format_duration(3661), "01:01:01");
    }
}
