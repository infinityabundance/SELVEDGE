//! Client construction, version 2: a builder replaces the constructor.

#[derive(Debug, Clone, PartialEq)]
pub struct Client {
    host: String,
    port: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    EmptyHost,
}

pub struct ClientBuilder {
    host: Option<String>,
    port: u16,
}

impl ClientBuilder {
    pub fn host(mut self, host: &str) -> Self {
        self.host = Some(host.to_string());
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn build(self) -> Result<Client, BuildError> {
        match self.host {
            Some(host) if !host.is_empty() => Ok(Client { host, port: self.port }),
            _ => Err(BuildError::EmptyHost),
        }
    }
}

pub fn builder() -> ClientBuilder {
    ClientBuilder { host: None, port: 80 }
}

impl Client {
    pub fn endpoint(&self, path: &str) -> String {
        format!("{}:{}{}", self.host, self.port, path)
    }
}
