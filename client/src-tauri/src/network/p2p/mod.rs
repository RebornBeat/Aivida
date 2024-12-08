mod session;
mod handler;

pub use session::P2PSession;
pub use handler::P2PHandler;

#[derive(Debug)]
pub struct P2PConfig {
    pub port: u16,
    pub max_connections: usize,
}
