//! Render bytes as decimal text. The Display impl on Vec<u8> violates
//! the orphan rule.

impl core::fmt::Display for Vec<u8> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = self
            .iter()
            .map(|byte| byte.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write!(formatter, "{text}")
    }
}

pub fn render(values: &[u8]) -> String {
    values.to_vec().to_string()
}
