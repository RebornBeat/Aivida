use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_utilization: f32,
    pub gpu_utilization: f32,
    pub memory_used: u64,
    pub network_bandwidth: NetworkBandwidth,
}

#[derive(Debug)]
pub struct SystemMetrics {
    pub cpu_usage: Vec<f32>,
    pub memory_usage: u64,
    pub gpu_metrics: Vec<GPUMetrics>,
    pub network_bandwidth: NetworkBandwidth,
}

#[derive(Debug)]
pub struct GPUMetrics {
    pub utilization: f32,
    pub memory_used: u64,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkBandwidth {
    pub upload: u64,
    pub download: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    pub execution_time: u64,
    pub resource_usage: ResourceUsage,
    pub cost: f64,
}
