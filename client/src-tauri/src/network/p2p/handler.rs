pub struct P2PHandler {
    encryption: Arc<ClientEncryption>,
    connections: Arc<RwLock<HashMap<Uuid, P2PSession>>>,
    listener: Option<TcpListener>,
}

impl P2PHandler {
    pub async fn init_listener(&self, port: u16) -> Result<(), AividaError> {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = TcpListener::bind(&addr).await
            .map_err(|e| AividaError::Network(format!("Failed to bind P2P listener: {}", e)))?;

        self.handle_connections(listener).await;
        Ok(())
    }

    async fn handle_connections(&self, listener: TcpListener) {
        let (tx, _) = broadcast::channel(100);
        let connections = self.connections.clone();
        let encryption = self.encryption.clone();

        tokio::spawn(async move {
            while let Ok((stream, peer_addr)) = listener.accept().await {
                let tx = tx.clone();
                let connections = connections.clone();
                let encryption = encryption.clone();

                tokio::spawn(async move {
                    if let Err(e) = Self::handle_connection(
                        stream,
                        peer_addr,
                        connections,
                        encryption,
                        tx
                    ).await {
                        eprintln!("P2P connection error: {}", e);
                    }
                });
            }
        });
    }

    async fn handle_connection(
        stream: TcpStream,
        peer_addr: SocketAddr,
        connections: Arc<RwLock<HashMap<Uuid, P2PSession>>>,
        encryption: Arc<ClientEncryption>,
        tx: broadcast::Sender<Message>,
    ) -> Result<(), AividaError> {
        // Handle handshake
        let (mut reader, mut writer) = stream.into_split();
        let mut handshake_buffer = vec![0u8; 1024];

        let n = reader.read(&mut handshake_buffer).await
            .map_err(|e| AividaError::Network(format!("Handshake read failed: {}", e)))?;

        let handshake: Message = serde_json::from_slice(&handshake_buffer[..n])
            .map_err(|e| AividaError::Network(format!("Invalid handshake: {}", e)))?;

        if !encryption.verify(&handshake.signature, &handshake.payload) {
            return Err(AividaError::Auth("Invalid handshake signature".into()));
        }

        // Create session
        let peer_id = Uuid::new_v4();
        let session = P2PSession::new(
            peer_id,
            TcpStream::from_std(stream.into_std().unwrap())?,
        );

        // Store session
        connections.write().await.insert(peer_id, session);

        // Handle messages
        Self::handle_messages(reader, encryption, tx, peer_id, connections.clone()).await?;

        connections.write().await.remove(&peer_id);
        Ok(())
    }

    async fn handle_messages(
        mut reader: OwnedReadHalf,
        encryption: Arc<ClientEncryption>,
        tx: broadcast::Sender<Message>,
        peer_id: Uuid,
        connections: Arc<RwLock<HashMap<Uuid, P2PSession>>>,
    ) -> Result<(), AividaError> {
        let mut buffer = vec![0u8; 8192];

        loop {
            let n = reader.read(&mut buffer).await?;
            if n == 0 {
                break;
            }

            let message: Message = serde_json::from_slice(&buffer[..n])?;
            if !encryption.verify(&message.signature, &message.payload) {
                continue;
            }

            // Update session metrics
            if let Some(session) = connections.write().await.get_mut(&peer_id) {
                session.metrics.bytes_received += n as u64;
                session.last_heartbeat = Instant::now();
            }

            tx.send(message)?;
        }

        Ok(())
    }

    pub async fn submit_job(&self, job: ComputeJob) -> Result<Uuid, AividaError> {
        // P2P job submission implementation here
        todo!("Implement P2P job submission")
    }

    async fn verify_peer(&self, peer_id: Uuid, public_key: &str) -> Result<bool, AividaError> {
        // Implement peer verification logic
        todo!()
    }

}
