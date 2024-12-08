use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use uuid::Uuid;
use crate::error::AividaError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRegistrationData {
    pub email: String,
    pub password: String,
    pub confirm_password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginCredentials {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResult {
    pub token: String,
    pub user_id: Uuid,
    pub expiration: chrono::DateTime<chrono::Utc>,
}

#[async_trait]
pub trait AuthProvider: Send + Sync {
    async fn register(&self, data: UserRegistrationData) -> Result<AuthResult, AividaError>;
    async fn login(&self, credentials: LoginCredentials) -> Result<AuthResult, AividaError>;
}
