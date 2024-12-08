#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub gpu_utilization: f32,
    pub cpu_utilization: f32,
    pub memory_used: u64,
}

#[derive(Debug, Clone)]
pub enum ResourceStatus {
    Available,
    Busy(f32),
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUpdate {
    resource_type: ResourceType,
    status: ResourceStatus,
    pub worker_id: Uuid,
    pub status: WorkerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    GPU(String),
    CPU(u32),
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
