use thiserror::Error;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Decryption error: {0}")]
    Decryption(String),

    #[error("Invalid key: {0}")]
    InvalidKey(String),

    #[error("Signature error: {0}")]
    SignatureError(String),

    #[error("Invalid data length: {0}")]
    InvalidLength(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}

impl From<CryptoError> for crate::AividaError {
    fn from(err: CryptoError) -> Self {
        match err {
            CryptoError::Encryption(msg) | CryptoError::Decryption(msg) => {
                Self::EncryptionError(msg)
            }
            CryptoError::SignatureError(msg) | CryptoError::VerificationFailed(msg) => {
                Self::AuthError(msg)
            }
            _ => Self::EncryptionError(err.to_string()),
        }
    }
}
