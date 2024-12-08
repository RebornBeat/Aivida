mod messages;
mod protocol;

pub use messages::C2SMessage;
pub use protocol::ServerProtocol;

#[derive(Debug)]
pub struct ServerConfig {
    pub addr: String,
    pub retry_policy: RetryPolicy,
}
