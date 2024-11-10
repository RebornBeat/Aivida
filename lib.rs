use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use ring::aead::{Aead, LessSafeKey, UnboundKey, AES_256_GCM};
use ring::rand::SystemRandom;
use uuid::Uuid;
use thiserror::Error;

// Error types for Aivida
#[derive(Error, Debug)]
pub enum AividaError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Resource allocation error: {0}")]
    ResourceError(String),
    #[error("Encryption error: {0}")]
    EncryptionError(String),
    #[error("Authentication error: {0}")]
    AuthError(String),
}

// Resource types that can be shared
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    GPU(String),  // GPU model
    CPU(u32),     // Number of cores
}

// Job status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed(String),
}

// Reward type selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardType {
    OffChainCredits,
    OnChainSolana,
    Hybrid,
}

// Core data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    id: Uuid,
    resources: Vec<ResourceType>,
    reward_type: RewardType,
    credits: f64,
    solana_wallet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeJob {
    id: Uuid,
    user_id: Uuid,
    resource_requirements: Vec<ResourceType>,
    status: JobStatus,
    created_at: chrono::DateTime<chrono::Utc>,
}

// Main Aivida node structure
pub struct AividaNode {
    users: Arc<Mutex<HashMap<Uuid, User>>>,
    jobs: Arc<Mutex<HashMap<Uuid, ComputeJob>>>,
    available_resources: Arc<Mutex<HashSet<(Uuid, ResourceType)>>>,
    encryption_key: LessSafeKey,
}

impl AividaNode {
    pub async fn new() -> Result<Self, AividaError> {
        // Initialize encryption
        let rng = SystemRandom::new();
        let unbound_key = UnboundKey::new(&AES_256_GCM, &[0u8; 32])
            .map_err(|e| AividaError::EncryptionError(e.to_string()))?;
        let encryption_key = LessSafeKey::new(unbound_key);

        Ok(Self {
            users: Arc::new(Mutex::new(HashMap::new())),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            available_resources: Arc::new(Mutex::new(HashSet::new())),
            encryption_key,
        })
    }

    // User management
    pub async fn register_user(&self, reward_type: RewardType, solana_wallet: Option<String>) -> Result<Uuid, AividaError> {
        let user_id = Uuid::new_v4();
        let user = User {
            id: user_id,
            resources: Vec::new(),
            reward_type,
            credits: 0.0,
            solana_wallet,
        };

        self.users.lock().unwrap().insert(user_id, user);
        Ok(user_id)
    }

    // Resource management
    pub async fn add_resource(&self, user_id: Uuid, resource: ResourceType) -> Result<(), AividaError> {
        let mut users = self.users.lock().unwrap();
        let user = users.get_mut(&user_id).ok_or(AividaError::AuthError("User not found".into()))?;
        user.resources.push(resource.clone());
        
        self.available_resources.lock().unwrap().insert((user_id, resource));
        Ok(())
    }

    // Job submission and management
    pub async fn submit_job(&self, user_id: Uuid, requirements: Vec<ResourceType>) -> Result<Uuid, AividaError> {
        let job_id = Uuid::new_v4();
        let job = ComputeJob {
            id: job_id,
            user_id,
            resource_requirements: requirements,
            status: JobStatus::Queued,
            created_at: chrono::Utc::now(),
        };

        self.jobs.lock().unwrap().insert(job_id, job);
        self.process_job_queue().await?;
        Ok(job_id)
    }

    async fn process_job_queue(&self) -> Result<(), AividaError> {
        let jobs = self.jobs.lock().unwrap();
        let available_resources = self.available_resources.lock().unwrap();

        // Simple job scheduling algorithm
        for (job_id, job) in jobs.iter() {
            if matches!(job.status, JobStatus::Queued) {
                // Check if required resources are available
                let mut can_process = true;
                for req in &job.resource_requirements {
                    let resource_available = available_resources.iter()
                        .any(|(_, r)| matches!(r, req));
                    if !resource_available {
                        can_process = false;
                        break;
                    }
                }

                if can_process {
                    // In a real implementation, we would allocate resources and start processing
                    tokio::spawn(self.clone().process_job(*job_id));
                }
            }
        }
        Ok(())
    }

    async fn process_job(self, job_id: Uuid) -> Result<(), AividaError> {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running;
            
            // Simulate job processing
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            
            job.status = JobStatus::Completed;
            self.distribute_rewards(job).await?;
        }
        Ok(())
    }

    // Reward distribution
    async fn distribute_rewards(&self, job: &ComputeJob) -> Result<(), AividaError> {
        let mut users = self.users.lock().unwrap();
        if let Some(user) = users.get_mut(&job.user_id) {
            match user.reward_type {
                RewardType::OffChainCredits => {
                    user.credits += 10.0; // Simple reward calculation
                }
                RewardType::OnChainSolana => {
                    if let Some(wallet) = &user.solana_wallet {
                        // Implement Solana reward distribution
                        println!("Distributing Solana rewards to wallet: {}", wallet);
                    }
                }
                RewardType::Hybrid => {
                    user.credits += 5.0;
                    if let Some(wallet) = &user.solana_wallet {
                        // Distribute hybrid rewards
                        println!("Distributing hybrid rewards to wallet: {}", wallet);
                    }
                }
            }
        }
        Ok(())
    }

    // Network handling
    pub async fn start_server(&self, addr: &str) -> Result<(), AividaError> {
        let listener = TcpListener::bind(addr).await
            .map_err(|e| AividaError::NetworkError(e.to_string()))?;

        println!("Aivida node listening on {}", addr);

        while let Ok((socket, _)) = listener.accept().await {
            tokio::spawn(self.handle_connection(socket));
        }

        Ok(())
    }

    async fn handle_connection(&self, socket: TcpStream) -> Result<(), AividaError> {
        // Implement connection handling, message processing, etc.
        Ok(())
    }

    // Encryption helpers
    fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, AividaError> {
        let nonce = ring::aead::Nonce::assume_unique_for_key([0u8; 12]);
        self.encryption_key.seal_in_place_append_tag(nonce, ring::aead::Aad::empty(), data)
            .map_err(|e| AividaError::EncryptionError(e.to_string()))
    }

    fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, AividaError> {
        let nonce = ring::aead::Nonce::assume_unique_for_key([0u8; 12]);
        self.encryption_key.open_in_place(nonce, ring::aead::Aad::empty(), encrypted_data)
            .map_err(|e| AividaError::EncryptionError(e.to_string()))
            .map(|data| data.to_vec())
    }
}

// Example usage and testing
#[tokio::main]
async fn main() -> Result<(), AividaError> {
    // Initialize node
    let node = AividaNode::new().await?;

    // Register a user
    let user_id = node.register_user(
        RewardType::Hybrid,
        Some("SolanaWalletAddress123".to_string())
    ).await?;

    // Add resources
    node.add_resource(user_id, ResourceType::GPU("NVIDIA RTX 3080".to_string())).await?;
    node.add_resource(user_id, ResourceType::CPU(8)).await?;

    // Submit a job
    let job_id = node.submit_job(
        user_id,
        vec![ResourceType::GPU("NVIDIA RTX 3080".to_string())]
    ).await?;

    // Start the server
    node.start_server("127.0.0.1:8080").await?;

    Ok(())
}
