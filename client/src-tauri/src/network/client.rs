use super::{p2p::P2PHandler, c2s::ServerProtocol};
use crate::resources::ResourceManager;

pub struct AividaClient {
    client_id: Uuid,
    p2p_handler: P2PHandler,
    server_protocol: ServerProtocol,
    resource_manager: ResourceManager,
    encryption: Arc<ClientEncryption>,
    worker_info: Option<WorkerInfo>,
    config: ClientConfig,
    auth_provider: Box<dyn AuthProvider>,
}

impl AividaClient {
    pub async fn new(config: ClientConfig) -> Result<Self, AividaError> {
        let encryption = Arc::new(ClientEncryption::new());
        let resource_manager = ResourceManager::new(Uuid::new_v4());
        let p2p_handler = P2PHandler::new(encryption.clone());
        let server_protocol = ServerProtocol::new(
            encryption.clone(),
            config.server_addr.clone(),
            resource_manager.clone(),
        );
        let auth_provider: Box<dyn AuthProvider> = match config.auth_mode {
            AuthMode::Server => Box::new(ServerAuth::new(server_protocol.clone())),
            AuthMode::DHT => Box::new(DHTAuth::new(dht_network.clone())),
        };

        Ok(Self {
            client_id: Uuid::new_v4(),
            encryption,
            p2p_handler,
            server_protocol,
            resource_manager,
            worker_info: None,
            config,
            auth_provider,
        })
    }

    pub async fn connect(&mut self) -> Result<(), AividaError> {
        // Connect to server first
        self.server_protocol.connect().await?;

        // If we're a worker, register capabilities
        if let Some(ref worker_info) = self.worker_info {
            self.register_as_worker(worker_info.clone()).await?;
        }

        // Initialize P2P if needed
        if self.config.security_level == SecurityLevel::Maximum {
            self.p2p_handler.init_listener(self.config.local_port.unwrap_or(0)).await?;
        }

        Ok(())
    }

    pub async fn register_user(&self, data: UserRegistrationData) -> Result<AuthResult, AividaError> {
        self.auth_provider.register(data).await
    }

    pub async fn login(&self, credentials: LoginCredentials) -> Result<AuthResult, AividaError> {
        self.auth_provider.login(credentials).await
    }

    pub async fn register_as_worker(&self, info: WorkerInfo) -> Result<(), AividaError> {
        // Validate worker info before attempting registration
        if let Some(ref worker_info) = self.worker_info {
            return Err(AividaError::Auth("Already registered as worker".into()));
        }
        // Pass to server protocol for registration
        self.server_protocol.register_worker(info).await
    }

    pub async fn submit_job(&self, job: ComputeJob) -> Result<Uuid, AividaError> {
        match (self.config.security_level, job.routing_mode) {
            (SecurityLevel::Maximum, ConnectionMode::P2P) => {
                self.p2p_handler.submit_job(job).await
            },
            _ => {
                self.server_protocol.submit_job(job).await
            }
        }
    }

    pub async fn rotate_encryption_keys(&mut self) -> Result<(), AividaError> {
        // Rotate local keys
        self.encryption.rotate_keys().await?;

        // Notify server of key rotation
        self.server_protocol.notify_key_rotation().await?;

        Ok(())
    }

    async fn handle_security_alert(&self, alert: SecurityAlert) -> Result<(), AividaError> {
        match alert {
            SecurityAlert::KeyCompromise => {
                self.rotate_encryption_keys().await?;
            },
            _ => {
                self.server_protocol.handle_security_alert(alert).await?;
            }
        }
        Ok(())
    }

    pub async fn get_worker_info(&self) -> Option<WorkerInfo> {
        if let Some(info) = &self.worker_info {
            Some(info.clone())
        } else {
            // Auto-detect capabilities if not set
            let capabilities = self.resource_manager.monitor.detect_hardware();
            Some(WorkerInfo {
                id: self.client_id,
                public_keys: self.encryption.get_public_keys(),
                location: self.config.location.clone(),
                capabilities,
                status: WorkerStatus::Available,
                security_level: self.config.security_level.clone(),
                connection_mode: self.config.connection_mode.clone(),
            })
        }
    }

    pub async fn update_resource_usage(&self, usage: ResourceUsage) -> Result<(), AividaError> {
        self.resource_manager.update_usage(usage.clone()).await?;
        self.server_protocol.send_resource_update(usage).await
    }
}
