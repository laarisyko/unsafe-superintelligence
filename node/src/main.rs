mod api;
mod bridge;
mod config;
mod consensus;
mod network;
mod scheduler;

use anyhow::Result;
use clap::Parser;
use config::{CliArgs, NodeConfig};
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    // Initialize logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .init();

    let config = NodeConfig::from(args);
    info!("Starting OpenClaw node on port {}", config.listen_port);

    // Ensure data directory exists.
    std::fs::create_dir_all(&config.data_dir)?;

    // Build and start the P2P swarm.
    let (mut swarm_driver, event_rx, command_tx) = network::SwarmDriver::new(&config).await?;

    // Try to connect to the Python engine bridge.
    let engine_bridge = bridge::EngineBridge::new(bridge::BridgeConfig::default());
    let engine_available = engine_bridge.health().await.unwrap_or(false);
    let engine_arc: Option<Arc<bridge::EngineBridge>> = if engine_available {
        info!("Connected to Python engine bridge on port {}", bridge::BridgeConfig::default().port);
        Some(Arc::new(engine_bridge))
    } else {
        info!("Python engine bridge not available; training will be deferred");
        None
    };

    // Shared state between scheduler and API.
    let api_shared_state = api::grpc_server::ApiSharedState::default();
    if engine_available {
        *api_shared_state.engine_connected.write().unwrap() = true;
    }

    // Start the scheduler that reacts to network events.
    let shard_map = consensus::shard_map::SharedShardMap::default();
    let reputation = consensus::shard_map::SharedReputationTable::default();
    let scheduler_handle = {
        let shard_map = shard_map.clone();
        let reputation = reputation.clone();
        let cmd_tx = command_tx.clone();
        let api_state = api_shared_state.clone();
        let engine_for_sched = engine_arc.clone();
        tokio::spawn(async move {
            scheduler::run(event_rx, cmd_tx, shard_map, reputation, engine_for_sched, api_state).await;
        })
    };

    // Start the gRPC API server.
    let api_handle = {
        let cmd_tx = command_tx.clone();
        let shard_map = shard_map.clone();
        let api_port = config.api_port;
        let engine_for_api = engine_arc.clone();
        let api_state = api_shared_state.clone();
        tokio::spawn(async move {
            if let Err(e) = api::grpc_server::run(api_port, cmd_tx, shard_map, engine_for_api, api_state).await {
                tracing::error!("gRPC server error: {}", e);
            }
        })
    };

    // Run the swarm (blocking).
    info!("Node is live. Press Ctrl-C to stop.");
    tokio::select! {
        res = swarm_driver.run() => {
            if let Err(e) = res {
                tracing::error!("Swarm driver error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Shutting down...");
        }
    }

    scheduler_handle.abort();
    api_handle.abort();
    info!("Node stopped.");
    Ok(())
}
