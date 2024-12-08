use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C2SMessage {
    pub message_type: C2SMessageType,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum C2SMessageType {
    JobSubmission(JobSubmissionMessage),
    ResourceUpdate(ResourceUpdateMessage),
    WorkerRegistration(WorkerRegistrationMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSubmissionMessage {
    pub job_id: Uuid,
    pub requester_id: Uuid,
    pub routing_mode: ConnectionMode,
    pub security_level: SecurityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdateMessage {
    pub worker_id: Uuid,
    pub status: WorkerStatus,
    pub current_load: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistrationMessage {
    pub worker_id: Uuid,
    pub capabilities: ResourceCapabilities,
    pub security_level: SecurityLevel,
}

impl C2SMessage {
    pub fn create<T: Serialize>(
        message_type: C2SMessageType,
        content: &T,
        encryption: &ClientEncryption,
    ) -> Result<Self, AividaError> {
        let payload = serde_json::to_vec(content)
            .map_err(|e| AividaError::Network(format!("Failed to serialize message: {}", e)))?;

        let signature = encryption.sign(&payload);

        Ok(Self {
            message_type,
            payload,
            signature,
        })
    }
}
