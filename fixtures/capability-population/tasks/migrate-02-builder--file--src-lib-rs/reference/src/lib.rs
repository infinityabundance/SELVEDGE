//! Connect and build an endpoint.

pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String> {
    client::builder()
        .host(host)
        .port(port)
        .build()
        .map(|client| client.endpoint(path))
        .map_err(|_| "empty host".to_string())
}
