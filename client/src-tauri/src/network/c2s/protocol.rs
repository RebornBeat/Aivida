pub struct ServerProtocol {
    encryption: Arc<ClientEncryption>,
    stream: Option<TcpStream>,
    active_jobs: Arc<RwLock<HashMap<Uuid, ComputeJob>>>,
    resource_manager: Arc<ResourceManager>,
}

impl ServerProtocol {
    pub async fn submit_job(&self, job: ComputeJob) -> Result<Uuid, AividaError> {
        let submission = JobSubmissionMessage {
            job_id: job.id,
            requester_id: job.requester_id,
            routing_mode: job.routing_mode.clone(),
            security_level: job.security_level.clone(),
        };

        let message = C2SMessage::create(
            C2SMessageType::JobSubmission(submission),
            &job,
            &self.encryption
        )?;

        self.send_message_with_response(&message).await?;

        self.active_jobs.write().await.insert(job.id, job);
        Ok(job.id)
    }

    pub async fn register_worker(&self, info: WorkerInfo) -> Result<(), AividaError> {
        let registration = WorkerRegistrationMessage {
            worker_id: info.id,
            capabilities: info.capabilities.clone(),
            security_level: info.security_level.clone(),
        };

        let message = self.create_message(
            C2SMessageType::WorkerRegistration(registration),
            &info
        )?;

        // Send registration and wait for response
        self.send_message_with_response(&message).await?;

        Ok(())
    }


    pub async fn send_resource_update(&self, usage: ResourceUsage) -> Result<(), AividaError> {
        let update = self.resource_manager.create_resource_update(&usage);

        let message = C2SMessage::create(
            C2SMessageType::ResourceUpdate(update),
            &usage,
            &self.encryption
        )?;

        self.send_message(&message).await
    }

    pub async fn handle_security_alert(&self, alert: SecurityAlert) -> Result<(), AividaError> {
        match alert {
            SecurityAlert::KeyCompromise => {
                // Notify client and trigger key rotation
                self.notify_key_compromise().await?;
            },
            SecurityAlert::UnauthorizedAccess(details) => {
                // Log and handle unauthorized access attempt
                self.handle_unauthorized_access(details).await?;
            },
            SecurityAlert::NetworkBreach(info) => {
                // Handle network security breach
                self.handle_network_breach(info).await?;
            }
        }
        Ok(())
    }

    async fn send_message(&self, message: Message) -> Result<(), AividaError> {
        if let Some(ref stream) = self.stream {
            let mut writer = stream.clone();
            writer.write_all(&serde_json::to_vec(&message)?)
                .await
                .map_err(|e| AividaError::Network(e.to_string()))?;
            Ok(())
        } else {
            Err(AividaError::Network("Not connected to server".into()))
        }
    }

    async fn send_message_with_response(&self, message: &C2SMessage) -> Result<Message, AividaError> {
        if let Some(ref stream) = self.stream {
            let (mut reader, mut writer) = stream.clone().into_split();

            // Send message
            writer.write_all(&serde_json::to_vec(message)?)
                .await
                .map_err(|e| AividaError::Network(format!("Failed to send message: {}", e)))?;

            // Wait for response
            let mut buffer = vec![0u8; 1024];
            let n = reader.read(&mut buffer)
                .await
                .map_err(|e| AividaError::Network(format!("Failed to read response: {}", e)))?;

            let response: Message = serde_json::from_slice(&buffer[..n])
                .map_err(|e| AividaError::Network(format!("Invalid response format: {}", e)))?;

            // Verify response
            if !self.encryption.verify(&response.signature, &response.payload) {
                return Err(AividaError::Auth("Invalid response signature".into()));
            }

            Ok(response)
        } else {
            Err(AividaError::Network("Not connected to server".into()))
        }
    }

}
