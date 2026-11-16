//! Connect and build an endpoint.

pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String> {
    client::builder()
        .host(host)
        .port(port)
        .build()
        .map(|client| client.endpoint(path))
        .map_err(|_| "empty host".to_string())
}

#[cfg(test)]
mod visible {
    use crate::connect;

    #[test]
    fn builds_an_endpoint() {
        assert_eq!(connect("example.org", 8080, "/x"), Ok("example.org:8080/x".to_string()));
    }

    #[test]
    fn rejects_an_empty_host() {
        assert_eq!(connect("", 80, "/"), Err("empty host".to_string()));
    }
}
