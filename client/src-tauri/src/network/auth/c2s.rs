use std::sync::Arc;
use super::types::{AuthProvider, AuthResult, LoginCredentials, UserRegistrationData};
use crate::error::AividaError;
use crate::network::c2s::protocol::ServerProtocol;
use crate::network::c2s::messages::{C2SMessage, C2SMessageType};
use async_trait::async_trait;

pub struct ServerAuth {
    protocol: Arc<ServerProtocol>,
}

impl ServerAuth {
    pub fn new(protocol: Arc<ServerProtocol>) -> Self {
        Self { protocol }
    }
}

#[async_trait]
impl AuthProvider for ServerAuth {
    async fn register(&self, data: UserRegistrationData) -> Result<AuthResult, AividaError> {
        let message = C2SMessage::create(
            C2SMessageType::UserRegistration(data.clone()),
            &data,
            &self.protocol.encryption
        )?;

        let response = self.protocol.send_message_with_response(&message).await?;

        serde_json::from_slice(&response.payload)
            .map_err(|e| AividaError::NetworkError(format!("Failed to parse auth result: {}", e)))
    }

    async fn login(&self, credentials: LoginCredentials) -> Result<AuthResult, AividaError> {
        let message = C2SMessage::create(
            C2SMessageType::UserLogin(credentials.clone()),
            &credentials,
            &self.protocol.encryption
        )?;

        let response = self.protocol.send_message_with_response(&message).await?;

        serde_json::from_slice(&response.payload)
            .map_err(|e| AividaError::NetworkError(format!("Failed to parse auth result: {}", e)))
    }
}
