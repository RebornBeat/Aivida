use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;
use sysinfo::System;
use crate::network::client::AividaClient;
use crate::types::*;

pub fn register_commands(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    app.register_command(tauri::generate_handler![
        init_client,
        submit_job,
        update_resources,
        register_user,
        login_user,
        get_resources,
        get_active_jobs,
        get_available_jobs,
        get_worker_info,
        detect_hardware,
    ])?;
    Ok(())
}

#[tauri::command]
async fn init_client(
    config: ClientConfig,
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<(), String> {
    let mut client_lock = client.lock().await;
    *client_lock = AividaClient::new(config).await
        .map_err(|e| e.to_string())?;

    client_lock.connect().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn submit_job(
    job: ComputeJob,
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<String, String> {
    let client_lock = client.lock().await;
    client_lock.submit_job(job).await
        .map(|uuid| uuid.to_string())
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn update_resources(
    usage: ResourceUsage,
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<(), String> {
    let client_lock = client.lock().await;
    client_lock.update_resource_usage(usage).await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn register_user(
    user_data: UserRegistrationData,
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<(), String> {
    let client_lock = client.lock().await;
    // Implement user registration
    if user_data.password != user_data.confirm_password {
        return Err("Passwords do not match".into());
    }

    // Hash password and store user
    todo!("Implement user registration");
}

#[tauri::command]
async fn login_user(
    credentials: LoginCredentials,
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<(), String> {
    let client_lock = client.lock().await;
    // Implement user authentication
    todo!("Implement user login");
}

#[tauri::command]
async fn get_resources(
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<ResourceMetrics, String> {
    let client_lock = client.lock().await;
    client_lock.get_current_metrics().await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_active_jobs(
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<Vec<Job>, String> {
    let client_lock = client.lock().await;
    let jobs = client_lock.active_jobs.read().await;
    Ok(jobs.values().cloned().collect())
}

#[tauri::command]
async fn get_available_jobs(
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<Vec<Job>, String> {
    let client_lock = client.lock().await;
    let jobs = client_lock.get_available_jobs().await
        .map_err(|e| e.to_string())?;
    Ok(jobs)
}

#[tauri::command]
async fn get_worker_info(
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<Option<WorkerInfo>, String> {
    let client_lock = client.lock().await;
    Ok(client_lock.get_worker_info().await)
}

#[tauri::command]
async fn detect_hardware(
    client: State<'_, Arc<Mutex<AividaClient>>>
) -> Result<ResourceCapabilities, String> {
    let client_lock = client.lock().await;
    Ok(client_lock.resource_manager.monitor.detect_hardware())
}
