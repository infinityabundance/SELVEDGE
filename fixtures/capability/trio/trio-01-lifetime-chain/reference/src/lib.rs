//! A table, an alias table, and a resolution chain.

pub type Table = [(&'static str, &'static str)];

pub fn lookup<'a>(table: &'a Table, key: &str) -> Option<&'a str> {
    table
        .iter()
        .find(|(name, _value)| *name == key)
        .map(|(_name, value)| *value)
}

pub fn resolve<'a>(
    table: &'a Table,
    aliases: &[(&'static str, &'static str)],
    key: &str,
) -> Option<&'a str> {
    if let Some(value) = lookup(table, key) {
        return Some(value);
    }
    aliases
        .iter()
        .find(|(alias, _target)| *alias == key)
        .and_then(|(_alias, target)| lookup(table, target))
}

pub fn chain<'a>(
    table: &'a Table,
    aliases: &[(&'static str, &'static str)],
    key: &str,
) -> Option<&'a str> {
    resolve(table, aliases, key)
}
