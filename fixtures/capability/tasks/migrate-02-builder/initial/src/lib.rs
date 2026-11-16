//! Connect and build an endpoint. This does not build: the client
//! constructor was removed in version 2.

pub fn connect(host: &str, port: u16, path: &str) -> Result<String, String> {
    let client = client::Client::new(host, port);
    Ok(client.endpoint(path))
}
