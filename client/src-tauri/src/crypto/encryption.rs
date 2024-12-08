use super::errors::AividaError;
use super::keys::{Keypair, PublicKey};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use std::sync::Arc;

const NONCE_LENGTH: usize = 24;

pub struct ClientEncryption {
    ml_kem_keypair: (Vec<u8>, Vec<u8>),
    dilithium_keypair: Keypair,
    chacha_key: Key,
    server_public_key: Option<Vec<u8>>,
}

impl ClientEncryption {
    pub fn new() -> Self {
        let mut rng = OsRng;

        // Generate main keypair
        let keypair = Arc::new(Keypair::generate());

        // Generate ChaCha20-Poly1305 key
        let mut chacha_key_data = [0u8; KEY_LENGTH];
        rng.fill_bytes(&mut chacha_key_data);
        let chacha_key = Key::from_slice(&chacha_key_data).clone();

        Self {
            keypair,
            chacha_key,
            server_public_key: None,
        }
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, AividaError> {
        // Create cipher instance
        let cipher = XChaCha20Poly1305::new(&self.chacha_key);

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_LENGTH];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);

        // Encrypt data
        cipher
            .encrypt(nonce, data)
            .map_err(|e| AividaError::EncryptionError(e.to_string()))
            .map(|ciphertext| {
                let mut result = Vec::with_capacity(NONCE_LENGTH + ciphertext.len());
                result.extend_from_slice(nonce.as_slice());
                result.extend(ciphertext);
                result
            })
    }

    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, AividaError> {
        if data.len() < NONCE_LENGTH {
            return Err(AividaError::EncryptionError(
                "Invalid encrypted data length".into(),
            ));
        }

        // Split nonce and ciphertext
        let nonce = XNonce::from_slice(&data[..NONCE_LENGTH]);
        let ciphertext = &data[NONCE_LENGTH..];

        // Create cipher and decrypt
        let cipher = XChaCha20Poly1305::new(&self.chacha_key);
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| AividaError::EncryptionError(e.to_string()))
    }

    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.keypair.sign(data)
    }

    pub fn verify(&self, signature: &[u8], data: &[u8]) -> bool {
        self.keypair.public.verify(signature, data)
    }

    pub fn set_server_public_key(&mut self, key: PublicKey) {
        self.server_public_key = Some(key);
    }

    pub fn get_public_key(&self) -> &PublicKey {
        &self.keypair.public
    }

    pub async fn rotate_keys(&mut self) -> Result<(), CryptoError> {
        // Rotate all keys
        self.dilithium_keypair.rotate().await?;

        // Generate new ChaCha key
        let mut new_key = [0u8; 32];
        OsRng.fill_bytes(&mut new_key);
        self.chacha_key = Key::from_slice(&new_key).clone();

        Ok(())
    }

}
