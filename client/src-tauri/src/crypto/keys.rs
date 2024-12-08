use blake3;
use rand::{rngs::OsRng, RngCore};

const KEY_LENGTH: usize = 32;
const SIGNATURE_LENGTH: usize = 64;

#[derive(Clone)]
pub struct Keypair {
    pub public: PublicKey,
    secret: SecretKey,
}

#[derive(Clone)]
pub struct PublicKey {
    key_data: Vec<u8>,
    key_hash: [u8; 32], // Blake3 hash of public key
}

#[derive(Clone)]
pub struct SecretKey {
    key_data: Vec<u8>,
}

impl Keypair {
    pub fn generate() -> Self {
        let mut rng = OsRng;

        // Generate secret key with high entropy
        let mut secret_data = vec![0u8; KEY_LENGTH];
        rng.fill_bytes(&mut secret_data);
        let secret = SecretKey {
            key_data: secret_data,
        };

        // Derive public key from secret key using Blake3
        let mut hasher = blake3::Hasher::new();
        hasher.update(&secret.key_data);
        let public_data = hasher.finalize().as_bytes().to_vec();

        // Create public key with precalculated hash
        let public = PublicKey::new(&public_data);

        Self { public, secret }
    }

    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        // Create signature context
        let mut hasher = blake3::Hasher::new();

        // Add secret key and data to context
        hasher.update(&self.secret.key_data);
        hasher.update(data);

        // Generate signature
        let signature = hasher.finalize();

        // Combine with public key hash for verification
        let mut final_signature = Vec::with_capacity(SIGNATURE_LENGTH);
        final_signature.extend_from_slice(signature.as_bytes());
        final_signature.extend_from_slice(&self.public.key_hash);

        final_signature
    }

    pub async fn rotate(&mut self) -> Result<(), CryptoError> {
        // Generate new keypair
        let new_keypair = Self::generate();

        // Update keys
        self.public = new_keypair.public;
        self.secret = new_keypair.secret;

        Ok(())
    }

}

impl PublicKey {
    pub fn new(key_data: &[u8]) -> Self {
        // Calculate key hash
        let mut hasher = blake3::Hasher::new();
        hasher.update(key_data);
        let key_hash = *hasher.finalize().as_bytes();

        Self {
            key_data: key_data.to_vec(),
            key_hash,
        }
    }

    pub fn verify(&self, signature: &[u8], data: &[u8]) -> bool {
        if signature.len() != SIGNATURE_LENGTH {
            return false;
        }

        // Extract signature components
        let sig_data = &signature[..32];
        let key_hash = &signature[32..];

        // Verify key hash matches
        if key_hash != self.key_hash {
            return false;
        }

        // Verify signature
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        let data_hash = hasher.finalize();

        constant_time_eq::constant_time_eq(sig_data, data_hash.as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.key_data
    }
}
