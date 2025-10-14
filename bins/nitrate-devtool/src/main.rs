use anyhow::Result;
use clap::Parser;
use tokio::time::{sleep, Duration};
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "nitrate-devtool", version)]
struct Cli {
    /// Path to a captured Stratum notify log (not implemented)
    #[arg(short, long)]
    _notify_log: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _cli = Cli::parse();
    info!("Devtool skeleton running. Implement replay/profiling here.");
    sleep(Duration::from_secs(1)).await;
    Ok(())
}