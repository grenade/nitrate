use anyhow::Result;
use clap::{Parser, Subcommand};
use nitrate_config::load_from_path;
use nitrate_core::DefaultEngine;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "miner", version)]
struct Cli {
    /// Path to config TOML
    #[arg(short, long, default_value = "miner.toml")]
    config: String,

    /// Log filter, e.g., info,debug,trace or crate=level
    #[arg(long)]
    log: Option<String>,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the miner (default)
    Run,
    /// List GPU devices
    Devices,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let filter = cli.log.or_else(|| std::env::var("RUST_LOG").ok()).unwrap_or_else(|| "info".into());
    tracing_subscriber::fmt().with_env_filter(EnvFilter::new(filter)).init();

    match cli.cmd.unwrap_or(Cmd::Run) {
        Cmd::Run => {
            let cfg = load_from_path(&cli.config)?;
            let engine = DefaultEngine::new(cfg).await?;
            engine.run().await?;
        }
        Cmd::Devices => {
            // To keep dependencies minimal, we call into the engine with dummy backend here.
            let cfg = load_from_path(&cli.config)?;
            let engine = DefaultEngine::new(cfg).await?;
            // enumerate happens in run(); here we just exit immediately.
            drop(engine);
        }
    }
    Ok(())
}