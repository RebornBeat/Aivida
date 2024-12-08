pub struct ResourceManager {
    monitor: ResourceMonitor,
    current_usage: Arc<Mutex<ResourceUsage>>,
    worker_id: Uuid,
}

impl ResourceManager {
    pub async fn update_usage(&self, usage: ResourceUsage) -> Result<(), AividaError> {
        let mut current = self.current_usage.lock().await;
        *current = usage;

        let status = if usage.cpu_utilization > 80.0 {
            WorkerStatus::Busy(usage.cpu_utilization)
        } else {
            WorkerStatus::Available
        };

        Ok(())
    }

    pub fn create_resource_update(&self, usage: &ResourceUsage) -> ResourceUpdate {
        ResourceUpdate {
            worker_id: self.worker_id,
            status: if usage.cpu_utilization > 80.0 {
                WorkerStatus::Busy(usage.cpu_utilization)
            } else {
                WorkerStatus::Available
            },
            current_load: usage.clone(),
        }
    }
}
