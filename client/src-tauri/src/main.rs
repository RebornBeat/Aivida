// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use tokio::sync::Mutex;
use client_lib::{AividaClient, register_commands};

fn main() {
    let config = ClientConfig {
        server_addr: "127.0.0.1:8080".to_string(),
        security_level: SecurityLevel::Standard,
        connection_mode: ConnectionMode::Standard,
        local_port: None,
        country_code: "US".to_string(),
    };

    let aivida_client = Arc::new(Mutex::new(
        AividaClient::new(config)
            .block_on()
            .expect("Failed to initialize client")
    ));

    tauri::Builder::default()
        .manage(aivida_client)
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            register_commands(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
