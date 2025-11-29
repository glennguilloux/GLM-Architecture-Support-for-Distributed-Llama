# INTELLECT-3 MoE Implementation Plan

## Overview
INTELLECT-3 is a 106B parameter Mixture-of-Experts (MoE) model that requires specialized distributed inference support. This document outlines the implementation strategy for efficient inference on consumer hardware.

## Architecture Specifications

### Model Parameters
- **Total Parameters**: 106B
- **Expert Count**: 16 experts (typical configuration)
- **Active Experts**: Top-2 gating (2 experts active per token)
- **Hidden Size**: 12,288
- **Attention Heads**: 96 (GQA with 32KV heads)
- **Layers**: 48 transformer layers
- **Expert Size**: ~6.6B parameters per expert

### MoE Architecture Details
```
INTELLECT-3 Architecture:
┌─────────────────────────────────────────────────────────┐
│ Input Embedding (12,288 dim)                            │
├─────────────────────────────────────────────────────────┤
│ 48x MoE Layers                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Expert Router (Gating Network)                      │ │
│ │ - Top-2 expert selection                            │ │
│ │ - Load balancing loss                               │ │
│ │ - Expert availability tracking                      │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ 16 Expert Networks                                  │ │
│ │ ┌──┐ ┌──┐ ┌──┐ ... ┌──┐                           │ │
│ │ │E1│ │E2│ │E3│     │E16│                          │ │
│ │ └──┘ └──┘ └──┘     └──┘                           │ │
│ │ Each: 6.6B parameters                              │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Final Layer Norm                                        │
├─────────────────────────────────────────────────────────┤
│ Output Projection (12,288 → vocab_size)                │
└─────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: Expert Router Implementation

```cpp
// src/glm/intellect-3.h - Main INTELLECT-3 architecture
#pragma once

#include "glm-4.h"
#include "intellect-router.h"

namespace intellect3 {

// INTELLECT-3 configuration
struct INTELLECT3Config {
    // Base model parameters
    uint32_t vocab_size = 150000;
    uint32_t hidden_size = 12288;
    uint32_t num_attention_heads = 96;
    uint32_t num_key_value_heads = 32; // GQA
    uint32_t num_hidden_layers = 48;
    uint32_t intermediate_size = 49152;
    
    // MoE specific parameters
    uint32_t num_experts = 16;
    uint32_t num_experts_active = 2; // Top-2 routing
    uint32_t expert_hidden_size = 12288;
    uint32_t expert_intermediate_size = 49152;
    
    // Load balancing
    float load_balance_alpha = 0.1;
    uint32_t capacity_factor = 1.25; // Token capacity per expert
    
    // Performance optimization
    bool enable_expert_caching = true;
    bool enable_dynamic_routing = true;
    uint32_t max_cache_size_gb = 8; // Per expert cache limit
};

// Expert network implementation
class Expert {
private:
    const INTELECT3Config& config;
    LayerNorm input_layer_norm;
    Linear gate_proj;     // Gating projection
    Linear up_proj;       // Up projection
    Linear down_proj;     // Down projection
    LayerNorm output_layer_norm;
    
    // Expert-specific weights
    std::vector<float> gate_weights;
    std::vector<float> up_weights;
    std::vector<float> down_weights;
    
public:
    explicit Expert(const INTELECT3Config& config);
    
    // Forward pass for single expert
    void forward(
        const float* input,
        float* output,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // Load expert weights
    bool load_weights(const std::string& weight_file);
    
    // Memory management
    size_t get_memory_usage() const;
    void clear_cache();
};

// Expert routing mechanism
class ExpertRouter {
private:
    const INTELECT3Config& config;
    Linear router_weights;
    
    // Load balancing
    std::vector<float> expert_load;
    std::vector<uint32_t> expert_usage_count;
    
    // Caching for frequently used experts
    std::unordered_map<uint64_t, std::vector<float>> expert_cache;
    std::mutex cache_mutex;
    
public:
    explicit ExpertRouter(const INTELECT3Config& config);
    
    // Compute expert routing decisions
    void route_tokens(
        const float* hidden_states,
        std::vector<std::vector<int32_t>>& expert_assignments,
        std::vector<std::vector<float>>& routing_weights,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // Update load balancing metrics
    void update_load_metrics(const std::vector<uint32_t>& expert_usage);
    
    // Get expert availability
    bool is_expert_available(uint32_t expert_id) const;
    
    // Memory-efficient expert loading
    void load_expert_on_demand(uint32_t expert_id);
    void unload_expert(uint32_t expert_id);
};

// Main INTELLECT-3 model class
class INTELLECT3Model {
private:
    const INTELECT3Config config;
    
    // Model components
    std::vector<Expert> experts;
    ExpertRouter router;
    std::vector<GLM4Attention> attention_layers;
    
    // Distributed inference support
    std::vector<uint32_t> local_experts; // Experts on this node
    std::vector<uint32_t> remote_experts; // Experts on other nodes
    
    // Communication
    std::unique_ptr<DistributedComm> comm;
    
public:
    explicit INTELECT3Model(const INTELECT3Config& config);
    
    // Forward pass with MoE
    void forward(
        const int32_t* input_ids,
        const float* attention_mask,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // Distributed inference
    void forward_distributed(
        const int32_t* input_ids,
        const float* attention_mask,
        uint32_t batch_size,
        uint32_t seq_len,
        const std::vector<uint32_t>& node_experts
    );
    
    // Expert management
    void setup_expert_distribution(const std::vector<std::vector<uint32_t>>& node_experts);
    void handle_remote_expert_request(uint32_t expert_id, const float* input, float* output);
    
    // Performance optimization
    void optimize_memory_layout();
    void enable_expert_caching(bool enable);
};

} // namespace intellect3
```

### Phase 2: Distributed MoE Implementation

```cpp
// src/glm/intellect-router.h - Expert routing implementation
#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include "../nn-network.h"

namespace intellect3 {

class ExpertRouter {
private:
    const INTELECT3Config& config;
    Linear router_gate; // Router gating network
    
    // Expert capacity management
    std::vector<uint32_t> expert_capacity;
    std::vector<float> expert_load_factors;
    
    // Top-K routing implementation
    void compute_top_k_routing(
        const float* gate_logits,
        std::vector<std::pair<int32_t, float>>& top_experts,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // Load balancing loss computation
    float compute_load_balancing_loss(
        const std::vector<uint32_t>& expert_usage,
        const std::vector<float>& routing_weights
    );
    
public:
    explicit ExpertRouter(const INTELECT3Config& config);
    
    // Main routing function
    void route_experts(
        const float* hidden_states,
        std::vector<std::vector<int32_t>>& expert_ids,
        std::vector<std::vector<float>>& expert_weights,
        std::vector<float>& load_balance_loss,
        uint32_t batch_size,
        uint32_t seq_len
    );
    
    // Expert capacity management
    bool check_expert_capacity(uint32_t expert_id, uint32_t token_count);
    void update_expert_usage(uint32_t expert_id, uint32_t token_count);
    
    // Adaptive routing based on load
    void adapt_routing_thresholds(float* thresholds, uint32_t num_experts);
    
    // Performance monitoring
    float get_expert_utilization(uint32_t expert_id) const;
    std::vector<float> get_all_utilizations() const;
};

} // namespace intellect3
```

### Phase 3: Consumer Hardware Optimization

```cpp
// src/glm/intellect-optimize.h - Memory and performance optimizations
#pragma once

#include "intellect-3.h"

namespace intellect3 {

// Memory optimization strategies
class MoEMemoryOptimizer {
public:
    // Expert weight compression
    static void compress_expert_weights(
        std::vector<float>& weights,
        CompressionType type,
        float compression_ratio
    );
    
    // Dynamic expert loading/unloading
    static void implement_expert_swapping(
        std::vector<Expert>& experts,
        uint32_t max_memory_gb
    );
    
    // KV cache optimization for MoE
    static void optimize_moe_kv_cache(
        float* kv_cache,
        uint32_t batch_size,
        uint32_t seq_len,
        uint32_t hidden_size,
        const std::vector<uint32_t>& active_experts
    );
    
    // Quantization for 106B model
    static void quantize_intellect3_model(
        INTELECT3Model& model,
        QuantizationConfig& config
    );
};

// Performance profiling
class MoEProfiler {
private:
    std::chrono::microseconds total_inference_time;
    std::chrono::microseconds expert_routing_time;
    std::chrono::microseconds expert_computation_time;
    std::chrono::microseconds communication_time;
    
    uint64_t total_tokens_processed;
    std::vector<uint64_t> expert_token_counts;
    
public:
    void start_profiling();
    void end_profiling();
    
    // Get performance metrics
    float get_tokens_per_second() const;
    float get_expert_efficiency() const;
    std::vector<float> get_expert_utilization_rates() const;
    
    // Memory usage
    size_t get_peak_memory_usage() const;
    void log_memory_usage();
};

// Consumer hardware configuration
struct ConsumerHardwareConfig {
    uint32_t gpu_memory_gb;
    uint32_t system_memory_gb;
    uint32_t gpu_count;
    bool enable_cpu_offloading;
    bool enable_memory_mapping;
    
    // Performance settings
    uint32_t max_batch_size;
    uint32_t max_seq_len;
    uint32_t preferred_expert_cache_size;
};

class HardwareOptimizer {
public:
    // Adapt model to available hardware
    static void adapt_to_hardware(
        INTELECT3Config& config,
        const ConsumerHardwareConfig& hw_config
    );
    
    // Determine optimal expert distribution
    static std::vector<std::vector<uint32_t>> optimize_expert_distribution(
        const INTELECT3Config& config,
        const ConsumerHardwareConfig& hw_config
    );
    
    // Enable memory optimization techniques
    static void enable_memory_optimizations(INTELECT3Model& model);
};

} // namespace intellect3
```

## Performance Targets

### Consumer Hardware Targets (RTX 3060 12GB)
- **Model Size**: 106B parameters
- **Memory Usage**: < 10GB VRAM (with quantization)
- **Inference Speed**: 5-10 tokens/second
- **Expert Caching**: 2-3 experts cached simultaneously

### Distributed Setup (4x Consumer GPUs)
- **Total Memory**: 40GB+ VRAM
- **Speedup**: 3-4x over single GPU
- **Scaling Efficiency**: 75%+
- **Memory per GPU**: 8-10GB

## Implementation Timeline

### Day 1-2: Expert Router
- Implement top-k expert selection
- Add load balancing mechanisms
- Test routing correctness

### Day 3-4: Expert Networks
- Implement expert forward pass
- Add weight loading/management
- Optimize expert computation

### Day 5-6: Distributed Support
- Add inter-node expert communication
- Implement dynamic expert loading
- Test distributed inference

### Day 7: Optimization & Testing
- Memory optimization for consumer hardware
- Performance profiling and tuning
- End-to-end testing

## Key Challenges & Solutions

### Challenge 1: Memory Management (106B parameters)
**Solution**: 
- 4-bit quantization: 53GB → 13.25GB
- Expert swapping: Load 2-3 experts at a time
- CPU offloading for inactive experts

### Challenge 2: Expert Load Balancing
**Solution**:
- Adaptive routing thresholds
- Dynamic capacity adjustment
- Load balancing loss in training

### Challenge 3: Distributed Communication
**Solution**:
- Expert request batching
- Async communication
- Local expert caching

### Challenge 4: Consumer Hardware Constraints
**Solution**:
- Aggressive quantization
- Memory-mapped weights
- CPU-GPU hybrid inference

This implementation plan provides a comprehensive approach to supporting the 106B INTELLECT-3 MoE model on consumer hardware through distributed inference.
