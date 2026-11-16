use failing_13_duration_minutes_field::format_duration;

#[test]
fn zero_is_all_zeroes() {
    assert_eq!(format_duration(0), "00:00:00");
}

#[test]
fn sub_minute_and_sub_hour_values() {
    assert_eq!(format_duration(59), "00:00:59");
    assert_eq!(format_duration(60), "00:01:00");
}

#[test]
fn an_hour_plus_one_second() {
    assert_eq!(format_duration(3661), "01:01:01");
}

#[test]
fn hours_exceed_a_day() {
    assert_eq!(format_duration(90061), "25:01:01");
}
