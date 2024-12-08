pub struct P2PSession {
    pub peer_id: Uuid,
    pub stream: TcpStream,
    pub established_at: Instant,
    pub last_heartbeat: Instant,
    pub metrics: SessionMetrics,
}

impl P2PSession {
    pub fn new(peer_id: Uuid, stream: TcpStream) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            stream,
            established_at: now,
            last_heartbeat: now,
            metrics: SessionMetrics::default(),
        }
    }
}
