use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// OpenClaw decentralized LLM network node.
#[derive(Parser, Debug, Clone)]
#[command(name = "openclaw-node", about = "Decentralized P2P node for LLM training & inference")]
pub struct CliArgs {
    /// Port to listen on for P2P connections.
    #[arg(short, long, default_value_t = 9000)]
    pub port: u16,

    /// Port for the gRPC API server.
    #[arg(long, default_value_t = 50051)]
    pub api_port: u16,

    /// Bootstrap peer multiaddress (e.g. /ip4/1.2.3.4/tcp/9000/p2p/12D3Koo...).
    /// Can be specified multiple times. Not required for the first node.
    #[arg(short, long)]
    pub bootstrap: Vec<String>,

    /// Path to store node data (keys, checkpoints, cached weights).
    #[arg(short, long, default_value = "./openclaw-data")]
    pub data_dir: PathBuf,

    /// Advertised GPU memory in MB (0 = CPU only).
    #[arg(long, default_value_t = 0)]
    pub gpu_memory_mb: u64,

    /// Advertised RAM in MB.
    #[arg(long, default_value_t = 4096)]
    pub ram_mb: u64,

    /// Accelerator type: cpu, cuda, rocm, tpu.
    #[arg(long, default_value = "cpu")]
    pub accelerator: String,

    /// Enable mDNS for LAN peer discovery.
    #[arg(long, default_value_t = true)]
    pub mdns: bool,

    /// Heartbeat interval in seconds.
    #[arg(long, default_value_t = 10)]
    pub heartbeat_interval_secs: u64,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

/// Runtime configuration derived from CLI args and environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub listen_port: u16,
    pub api_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub data_dir: PathBuf,
    pub gpu_memory_mb: u64,
    pub ram_mb: u64,
    pub cpu_cores: u32,
    pub accelerator: String,
    pub mdns_enabled: bool,
    pub heartbeat_interval_secs: u64,
}

impl From<CliArgs> for NodeConfig {
    fn from(args: CliArgs) -> Self {
        let cpu_cores = num_cpus();
        Self {
            listen_port: args.port,
            api_port: args.api_port,
            bootstrap_peers: args.bootstrap,
            data_dir: args.data_dir,
            gpu_memory_mb: args.gpu_memory_mb,
            ram_mb: args.ram_mb,
            cpu_cores,
            accelerator: args.accelerator,
            mdns_enabled: args.mdns,
            heartbeat_interval_secs: args.heartbeat_interval_secs,
        }
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}
