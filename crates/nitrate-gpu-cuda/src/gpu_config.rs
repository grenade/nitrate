use std::collections::HashMap;

/// GPU-specific configuration for optimal mining performance
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Number of blocks in the grid
    pub grid_size: u32,
    /// Number of threads per block
    pub block_size: u32,
    /// Number of nonces each thread processes per kernel launch
    pub nonces_per_thread: u32,
    /// Size of the result ring buffer
    pub ring_capacity: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            grid_size: 512,
            block_size: 256,
            nonces_per_thread: 1,
            ring_capacity: 8192,
        }
    }
}

/// Database of optimal configurations for known GPU models
#[derive(Clone, Debug)]
pub struct GpuDatabase {
    configs: HashMap<String, GpuConfig>,
}

impl GpuDatabase {
    pub fn new() -> Self {
        let mut configs = HashMap::new();

        // RTX 5090: 21,760 CUDA cores, 170 SMs
        // Use more aggressive settings for this high-end GPU
        configs.insert(
            "NVIDIA GeForce RTX 5090".to_string(),
            GpuConfig {
                grid_size: 5440,      // 170 SMs * 32 blocks per SM (double occupancy)
                block_size: 512,      // Optimal for Blackwell architecture
                nonces_per_thread: 8, // Process 8 nonces per thread for better throughput
                ring_capacity: 65536, // Very large buffer for massive throughput
            },
        );

        // RTX 4090: 16,384 CUDA cores, 128 SMs
        configs.insert(
            "NVIDIA GeForce RTX 4090".to_string(),
            GpuConfig {
                grid_size: 2048, // 128 SMs * 16 blocks per SM
                block_size: 512, // Optimal for Ada Lovelace
                nonces_per_thread: 4,
                ring_capacity: 16384,
            },
        );

        // RTX 3090: 10,496 CUDA cores, 82 SMs
        configs.insert(
            "NVIDIA GeForce RTX 3090".to_string(),
            GpuConfig {
                grid_size: 1312, // 82 SMs * 16 blocks per SM
                block_size: 512,
                nonces_per_thread: 2,
                ring_capacity: 16384,
            },
        );

        // RTX 3080: 8,704 CUDA cores, 68 SMs
        configs.insert(
            "NVIDIA GeForce RTX 3080".to_string(),
            GpuConfig {
                grid_size: 1088, // 68 SMs * 16 blocks per SM
                block_size: 512,
                nonces_per_thread: 2,
                ring_capacity: 8192,
            },
        );

        // RTX 3070: 5,888 CUDA cores, 46 SMs
        configs.insert(
            "NVIDIA GeForce RTX 3070".to_string(),
            GpuConfig {
                grid_size: 736, // 46 SMs * 16 blocks per SM
                block_size: 256,
                nonces_per_thread: 2,
                ring_capacity: 8192,
            },
        );

        // RTX 3060: 3,584 CUDA cores, 28 SMs
        configs.insert(
            "NVIDIA GeForce RTX 3060".to_string(),
            GpuConfig {
                grid_size: 448, // 28 SMs * 16 blocks per SM
                block_size: 256,
                nonces_per_thread: 1,
                ring_capacity: 4096,
            },
        );

        // RTX 3060 Ti: 4,864 CUDA cores, 38 SMs
        configs.insert(
            "NVIDIA GeForce RTX 3060 Ti".to_string(),
            GpuConfig {
                grid_size: 608, // 38 SMs * 16 blocks per SM
                block_size: 256,
                nonces_per_thread: 2,
                ring_capacity: 8192,
            },
        );

        // Older high-end cards
        configs.insert(
            "NVIDIA GeForce RTX 2080 Ti".to_string(),
            GpuConfig {
                grid_size: 544, // 68 SMs * 8 blocks per SM (Turing)
                block_size: 256,
                nonces_per_thread: 1,
                ring_capacity: 4096,
            },
        );

        // Datacenter cards
        configs.insert(
            "NVIDIA A100".to_string(),
            GpuConfig {
                grid_size: 3456, // 108 SMs * 32 blocks per SM
                block_size: 512,
                nonces_per_thread: 8,
                ring_capacity: 65536,
            },
        );

        configs.insert(
            "NVIDIA H100".to_string(),
            GpuConfig {
                grid_size: 4352, // 136 SMs * 32 blocks per SM
                block_size: 512,
                nonces_per_thread: 8,
                ring_capacity: 65536,
            },
        );

        Self { configs }
    }

    /// Get optimized configuration for a GPU by name
    pub fn get_config(&self, gpu_name: &str) -> GpuConfig {
        // Try exact match first
        if let Some(config) = self.configs.get(gpu_name) {
            return config.clone();
        }

        // Try partial matches - check for key GPU model identifiers
        // Priority order: check for most specific models first
        let gpu_lower = gpu_name.to_lowercase();

        // Check for RTX 5090 specifically (highest priority)
        if gpu_lower.contains("5090") || gpu_lower.contains("rtx 5090") {
            if let Some(config) = self.configs.get("NVIDIA GeForce RTX 5090") {
                return config.clone();
            }
        }

        // Then check other models
        for (key, config) in &self.configs {
            let key_lower = key.to_lowercase();
            // Extract model number (e.g., "5090", "4090", "3090")
            if let Some(model) = key_lower.split("rtx").nth(1) {
                let model = model.trim();
                if gpu_lower.contains(model) {
                    return config.clone();
                }
            }
        }

        // Fallback to defaults
        GpuConfig::default()
    }
}

impl Default for GpuDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_database() {
        let db = GpuDatabase::new();

        // Test exact match
        let config = db.get_config("NVIDIA GeForce RTX 5090");
        assert_eq!(config.grid_size, 2720);
        assert_eq!(config.block_size, 512);

        // Test partial match
        let config = db.get_config("RTX 3060");
        assert_eq!(config.grid_size, 448);

        // Test unknown GPU
        let config = db.get_config("Unknown GPU");
        assert_eq!(config.grid_size, 512); // Should return defaults
    }
}
