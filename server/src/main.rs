use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use chacha20poly1305::{XChaCha20Poly1305, Key, XNonce};
use chacha20poly1305::aead::{Aead, NewAead};
use ml_kem::{PRIVATE_KEY_LENGTH, PUBLIC_KEY_LENGTH};
use dilithium::{Keypair, PublicKey, SecretKey};
use rand::{rngs::OsRng, RngCore};
use sqlx::{Pool, Postgres};
use sqlx::postgres::PgPoolOptions;
use blake3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// Error types
#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Encryption error: {0}")]
    Encryption(String),
    #[error("Authentication error: {0}")]
    Auth(String),
    #[error("Job processing error: {0}")]
    Job(String),
}

// Security and Connection Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityLevel {
    Maximum,
    Standard,
    Basic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionMode {
    P2P,
    Standard,
}

// Message Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    WorkerRegistration(WorkerInfo),
    JobSubmission(ComputeJob),
    JobResult(JobResult),
    ResourceUpdate(ResourceUpdate),
    P2PRequest(P2PRequest),
    CreditTransaction(CreditTransaction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerMessage {
    pub message_type: MessageType,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

// Worker and Resource Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: Uuid,
    pub public_keys: PublicKeys,
    pub location: GeoLocation,
    pub capabilities: ResourceCapabilities,
    pub status: WorkerStatus,
    pub security_level: SecurityLevel,
    pub connection_mode: ConnectionMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKeys {
    pub mlkem_public_key: String,
    pub dilithium_public_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub country_code: String,
    pub region: String,
    pub security_requirements: Vec<SecurityRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    pub gpu: Vec<GPUInfo>,
    pub cpu: CPUInfo,
    pub memory: u64,
    pub bandwidth: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUInfo {
    pub model: String,
    pub memory: u64,
    pub compute_capability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUInfo {
    pub cores: u32,
    pub threads: u32,
    pub architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerStatus {
    Available,
    Busy(f32),
    Offline,
    Maintenance,
}

// Security Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRequirement {
    pub regulation_type: RegulationType,
    pub storage_required: bool,
    pub encryption_level: SecurityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegulationType {
    GDPR,
    CCPA,
    HIPAA,
    Standard,
}

// Job Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeJob {
    pub id: Uuid,
    pub requester_id: Uuid,
    pub worker_id: Option<Uuid>,
    pub requirements: JobRequirements,
    pub status: JobStatus,
    pub security_level: SecurityLevel,
    pub routing_mode: ConnectionMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequirements {
    pub min_gpu_memory: u64,
    pub min_cpu_cores: u32,
    pub min_memory: u64,
    pub required_location: Option<String>,
    pub security_requirements: Vec<SecurityRequirement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Assigned,
    Processing,
    Completed,
    Failed(String),
}

// Result and Metrics Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: Uuid,
    pub result: Vec<u8>,
    pub metrics: JobMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    pub execution_time: u64,
    pub resource_usage: ResourceUsage,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub gpu_utilization: f32,
    pub cpu_utilization: f32,
    pub memory_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdate {
    pub worker_id: Uuid,
    pub status: WorkerStatus,
    pub current_load: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PRequest {
    pub requester_id: Uuid,
    pub worker_id: Uuid,
    pub job_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditTransaction {
    pub id: Uuid,
    pub user_id: Uuid,
    pub amount: f64,
    pub transaction_type: TransactionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    JobPayment(Uuid),
    ResourceProvision(Uuid),
    Deposit,
    Withdrawal,
}

// Quantum Encryption Manager
struct QuantumEncryption {
    ml_kem_keypair: (Vec<u8>, Vec<u8>),
    dilithium_keypair: Keypair,
    chacha_key: Key,
}

impl QuantumEncryption {
    fn new() -> Self {
        let mut rng = OsRng;

        let mut private_key = vec![0u8; PRIVATE_KEY_LENGTH];
        let mut public_key = vec![0u8; PUBLIC_KEY_LENGTH];
        rng.fill_bytes(&mut private_key);
        rng.fill_bytes(&mut public_key);

        let dilithium_keypair = Keypair::generate(&mut rng);

        let mut chacha_key = [0u8; 32];
        rng.fill_bytes(&mut chacha_key);

        Self {
            ml_kem_keypair: (private_key, public_key),
            dilithium_keypair,
            chacha_key: Key::from_slice(&chacha_key).clone(),
        }
    }

    fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, ServerError> {
        let cipher = XChaCha20Poly1305::new(&self.chacha_key);
        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut nonce);
        let nonce = XNonce::from_slice(&nonce);

        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| ServerError::Encryption(e.to_string()))?;
        let mut result = nonce.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, ServerError> {
        let cipher = XChaCha20Poly1305::new(&self.chacha_key);
        let nonce = XNonce::from_slice(&data[..24]);
        let ciphertext = &data[24..];
        cipher.decrypt(nonce, ciphertext)
            .map_err(|e| ServerError::Encryption(e.to_string()))
    }

    fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.dilithium_keypair.sign(data)
    }

    fn verify(&self, signature: &[u8], data: &[u8]) -> bool {
        self.dilithium_keypair.public.verify(signature, data)
    }
}

// Server State Management
struct ServerState {
    encryption: Arc<QuantumEncryption>,
    db: Pool<Postgres>,
    workers: Arc<Mutex<HashMap<Uuid, WorkerInfo>>>,
    jobs: Arc<Mutex<HashMap<Uuid, ComputeJob>>>,
    p2p_sessions: Arc<Mutex<HashMap<Uuid, P2PSession>>>,
}

struct P2PSession {
    requester_id: Uuid,
    worker_id: Uuid,
    job_id: Uuid,
    established_at: SystemTime,
    status: P2PSessionStatus,
}

#[derive(Debug, Clone)]
enum P2PSessionStatus {
    Negotiating,
    Active,
    Completed,
    Failed(String),
}

impl ServerState {
    async fn new(database_url: &str) -> Result<Self, ServerError> {
        let db = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        Ok(Self {
            encryption: Arc::new(QuantumEncryption::new()),
            db,
            workers: Arc::new(Mutex::new(HashMap::new())),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            p2p_sessions: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    async fn handle_message(&self, message: ServerMessage) -> Result<ServerMessage, ServerError> {
        if !self.encryption.verify(&message.signature, &message.payload) {
            return Err(ServerError::Auth("Invalid signature".into()));
        }

        match message.message_type {
            MessageType::WorkerRegistration(info) => self.handle_worker_registration(info).await,
            MessageType::JobSubmission(job) => self.handle_job_submission(job).await,
            MessageType::JobResult(result) => self.handle_job_result(result).await,
            MessageType::ResourceUpdate(update) => self.handle_resource_update(update).await,
            MessageType::P2PRequest(request) => self.handle_p2p_request(request).await,
            MessageType::CreditTransaction(tx) => self.handle_credit_transaction(tx).await,
        }
    }

    async fn store_metrics(&self, worker_id: Uuid, metrics: &ResourceUsage) -> Result<(), ServerError> {
        sqlx::query!(
            r#"
            INSERT INTO worker_metrics (id, worker_id, cpu_utilization, gpu_utilization, memory_used)
            VALUES ($1, $2, $3, $4, $5)
            "#,
            Uuid::new_v4(),
            worker_id,
            metrics.cpu_utilization,
            metrics.gpu_utilization,
            metrics.memory_used as i64
        )
        .execute(&self.db)
        .await?;

        Ok(())
    }

    // Handler Implementations
    async fn handle_worker_registration(&self, worker: WorkerInfo) -> Result<ServerMessage, ServerError> {
        // Store worker info in database
        sqlx::query!(
            r#"
            INSERT INTO workers (
                id, mlkem_public_key, dilithium_public_key,
                country_code, region, security_level,
                connection_mode, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE
            SET mlkem_public_key = EXCLUDED.mlkem_public_key,
                dilithium_public_key = EXCLUDED.dilithium_public_key,
                status = EXCLUDED.status
            "#,
            worker.id,
            worker.public_keys.mlkem_public_key,
            worker.public_keys.dilithium_public_key,
            worker.location.country_code,
            worker.location.region,
            format!("{:?}", worker.security_level),
            format!("{:?}", worker.connection_mode),
            format!("{:?}", worker.status),
        )
        .execute(&self.db)
        .await?;

        // Store capabilities
        sqlx::query!(
            r#"
            INSERT INTO worker_capabilities (worker_id, capabilities)
            VALUES ($1, $2)
            ON CONFLICT (worker_id) DO UPDATE
            SET capabilities = EXCLUDED.capabilities
            "#,
            worker.id,
            serde_json::to_value(&worker.capabilities)?
        )
        .execute(&self.db)
        .await?;

        // Update in-memory state
        self.workers.lock().await.insert(worker.id, worker.clone());

        // Create response
        let response_payload = serde_json::to_vec(&"Registration successful")?;
        let signature = self.encryption.sign(&response_payload);

        Ok(ServerMessage {
            message_type: MessageType::WorkerRegistration(worker),
            payload: response_payload,
            signature,
        })
    }

    async fn handle_job_submission(&self, job: ComputeJob) -> Result<ServerMessage, ServerError> {
        let job_id = job.id;

        // If Maximum security level and P2P mode, handle differently
        if job.security_level == SecurityLevel::Maximum && job.routing_mode == ConnectionMode::P2P {
            // Find suitable worker
            let worker_id = self.find_suitable_worker(&job).await?
                .ok_or_else(|| ServerError::Job("No suitable worker found".into()))?;

            // Initialize P2P session
            let session = P2PSession {
                requester_id: job.requester_id,
                worker_id,
                job_id,
                established_at: SystemTime::now(),
                status: P2PSessionStatus::Negotiating,
            };

            self.p2p_sessions.lock().await.insert(job_id, session);

            // Store job metadata (without actual data for P2P jobs)
            sqlx::query!(
                r#"
                INSERT INTO jobs (
                    id, requester_id, worker_id, status,
                    security_level, routing_mode, requirements
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                "#,
                job_id,
                job.requester_id,
                Some(worker_id),
                "Pending",
                format!("{:?}", job.security_level),
                format!("{:?}", job.routing_mode),
                serde_json::to_value(&job.requirements)?
            )
            .execute(&self.db)
            .await?;

            // Return P2P connection details
            let worker = self.workers.lock().await.get(&worker_id)
                .ok_or_else(|| ServerError::Job("Worker not found".into()))?.clone();

            let p2p_info = serde_json::to_vec(&(worker_id, worker.public_keys))?;
            let signature = self.encryption.sign(&p2p_info);

            Ok(ServerMessage {
                message_type: MessageType::P2PRequest(P2PRequest {
                    requester_id: job.requester_id,
                    worker_id,
                    job_id,
                }),
                payload: p2p_info,
                signature,
            })
        } else {
            // Handle standard job submission
            let worker_id = self.find_suitable_worker(&job).await?
                .ok_or_else(|| ServerError::Job("No suitable worker found".into()))?;

            // Store full job details
            sqlx::query!(
                r#"
                INSERT INTO jobs (
                    id, requester_id, worker_id, status,
                    security_level, routing_mode, requirements
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                "#,
                job_id,
                job.requester_id,
                Some(worker_id),
                "Pending",
                format!("{:?}", job.security_level),
                format!("{:?}", job.routing_mode),
                serde_json::to_value(&job.requirements)?
            )
            .execute(&self.db)
            .await?;

            self.jobs.lock().await.insert(job_id, job.clone());

            // Create response
            let response_payload = serde_json::to_vec(&format!("Job {} submitted successfully", job_id))?;
            let signature = self.encryption.sign(&response_payload);

            Ok(ServerMessage {
                message_type: MessageType::JobSubmission(job),
                payload: response_payload,
                signature,
            })
        }
    }

    async fn handle_job_result(&self, result: JobResult) -> Result<ServerMessage, ServerError> {
        // Update job status
        sqlx::query!(
            r#"
            UPDATE jobs
            SET status = $1, completed_at = CURRENT_TIMESTAMP,
                result_metrics = $2
            WHERE id = $3
            "#,
            "Completed",
            serde_json::to_value(&result.metrics)?,
            result.job_id
        )
        .execute(&self.db)
        .await?;

        // Store metrics
        if let Some(job) = self.jobs.lock().await.get(&result.job_id) {
            if let Some(worker_id) = job.worker_id {
                self.store_metrics(worker_id, &result.metrics.resource_usage).await?;
            }
        }

        // Handle P2P session completion if applicable
        if let Some(session) = self.p2p_sessions.lock().await.get_mut(&result.job_id) {
            session.status = P2PSessionStatus::Completed;
        }

        // Create response
        let response_payload = serde_json::to_vec(&"Result processed successfully")?;
        let signature = self.encryption.sign(&response_payload);

        Ok(ServerMessage {
            message_type: MessageType::JobResult(result),
            payload: response_payload,
            signature,
        })
    }

    async fn handle_resource_update(&self, update: ResourceUpdate) -> Result<ServerMessage, ServerError> {
        // Store metrics
        self.store_metrics(update.worker_id, &update.current_load).await?;

        // Update worker status
        self.update_worker_status(update.worker_id, &update.status).await?;

        // Update in-memory state
        if let Some(worker) = self.workers.lock().await.get_mut(&update.worker_id) {
            worker.status = update.status.clone();
        }

        // Create response
        let response_payload = serde_json::to_vec(&"Resource update processed")?;
        let signature = self.encryption.sign(&response_payload);

        Ok(ServerMessage {
            message_type: MessageType::ResourceUpdate(update),
            payload: response_payload,
            signature,
        })
    }

    async fn handle_p2p_request(&self, request: P2PRequest) -> Result<ServerMessage, ServerError> {
        // Verify both parties exist and are available
        let workers = self.workers.lock().await;
        let worker = workers.get(&request.worker_id)
            .ok_or_else(|| ServerError::Job("Worker not found".into()))?;

        // Verify security levels are compatible
        if worker.security_level != SecurityLevel::Maximum {
            return Err(ServerError::Job("Worker security level insufficient for P2P".into()));
        }

        // Create or update P2P session
        let session = P2PSession {
            requester_id: request.requester_id,
            worker_id: request.worker_id,
            job_id: request.job_id,
            established_at: SystemTime::now(),
            status: P2PSessionStatus::Active,
        };

        self.p2p_sessions.lock().await.insert(request.job_id, session);

        // Create response with connection details
        let response_data = serde_json::to_vec(&worker.public_keys)?;
        let signature = self.encryption.sign(&response_data);

        Ok(ServerMessage {
            message_type: MessageType::P2PRequest(request),
            payload: response_data,
            signature,
        })
    }

    async fn handle_credit_transaction(&self, tx: CreditTransaction) -> Result<ServerMessage, ServerError> {
        // Store transaction
        sqlx::query!(
            r#"
            INSERT INTO credit_transactions (
                id, user_id, amount, transaction_type,
                created_at
            )
            VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
            "#,
            tx.id,
            tx.user_id,
            tx.amount,
            format!("{:?}", tx.transaction_type)
        )
        .execute(&self.db)
        .await?;

        // Update user balance
        sqlx::query!(
            r#"
            UPDATE users
            SET credit_balance = credit_balance + $1
            WHERE id = $2
            "#,
            tx.amount,
            tx.user_id
        )
        .execute(&self.db)
        .await?;

        // Create response
        let response_payload = serde_json::to_vec(&"Transaction processed")?;
        let signature = self.encryption.sign(&response_payload);

        Ok(ServerMessage {
            message_type: MessageType::CreditTransaction(tx),
            payload: response_payload,
            signature,
        })
    }

}

// Connection handling
async fn handle_connection(socket: TcpStream, state: Arc<ServerState>) -> Result<(), ServerError> {
    let (mut reader, mut writer) = socket.into_split();
    let mut buf = vec![0u8; 1024];

    loop {
        let n = reader.read(&mut buf).await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        if n == 0 {
            break;
        }

        let message: ServerMessage = serde_json::from_slice(&buf[..n])
            .map_err(|e| ServerError::Network(e.to_string()))?;

        let response = state.handle_message(message).await?;

        let response_data = serde_json::to_vec(&response)
            .map_err(|e| ServerError::Network(e.to_string()))?;

        writer.write_all(&response_data).await
            .map_err(|e| ServerError::Network(e.to_string()))?;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/aivida".to_string());

    let state = Arc::new(ServerState::new(&database_url).await?);
    let addr = "0.0.0.0:8080";

    println!("Starting Aivida server on {}", addr);
    let listener = TcpListener::bind(addr).await?;

    while let Ok((socket, _)) = listener.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(socket, state).await {
                eprintln!("Connection error: {}", e);
            }
        });
    }

    Ok(())
}
