use crate::crypto::PublicKeys;
use crate::resource::ResourceCapabilities;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
pub enum WorkerStatus {
    Available,
    Busy(f32), // Load percentage
    Offline,
    Maintenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub country_code: String,
    pub region: String,
    pub security_requirements: Vec<SecurityRequirement>,
}
