#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: Uuid,
    pub result: Vec<u8>,
    pub metrics: JobMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Maximum,  // P2P enabled, full encryption
    Standard, // Server routing with encryption
    Basic,    // Server routing with basic security
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionMode {
    P2P,
    Standard,
}
