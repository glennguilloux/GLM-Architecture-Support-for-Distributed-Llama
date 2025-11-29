/*
 * GLM GPU Kernels for Distributed Llama
 * Optimized CUDA kernels for GLM-4 and INTELLECT-3 inference
 * Author: Glenn Guilloux
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Constants for GPU optimization
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int VECTOR_SIZE = 4;

// GLM-4 specific constants
constexpr int GLM_ROPE_DIM = 64;
constexpr int GLM_BLOCK_SIZE = 512;

// INTELLECT-3 MoE constants
constexpr int MAX_EXPERTS = 16;
constexpr int ACTIVE_EXPERTS = 2;
constexpr int MOE_CAPACITY = 256;

// Utility functions for CUDA
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[warp] = val;
    }
    
    __syncthreads();
    
    val = (threadIdx.x < WARP_SIZE) ? shared[lane] : 0;
    
    if (warp == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// GLM-4 Attention Kernel Optimizations
__global__ void glm4_attention_forward_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ attention_mask,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Bidirectional attention for GLM-4
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Process each attention head
    for (int head = blockIdx.y; head < num_heads; head += gridDim.y) {
        for (int seq_pos = blockIdx.z; seq_pos < seq_len; seq_pos += gridDim.z) {
            for (int idx = tid; idx < seq_len * head_dim; idx += total_threads) {
                int q_seq = idx / head_dim;
                int dim = idx % head_dim;
                
                if (q_seq < seq_len && dim < head_dim) {
                    // GLM-4 bidirectional attention computation
                    float attention_sum = 0.0f;
                    
                    #pragma unroll
                    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
                        // Apply attention mask for GLM
                        if (attention_mask[q_seq * seq_len + k_seq] > 0.0f) {
                            // Compute attention score
                            float q_val = query[q_seq * num_heads * head_dim + head * head_dim + dim];
                            float k_val = key[k_seq * num_heads * head_dim + head * head_dim + dim];
                            float score = q_val * k_val / sqrtf(head_dim);
                            
                            // Apply GLM-specific attention pattern
                            attention_sum += score;
                        }
                    }
                    
                    // Softmax and value aggregation
                    float output_val = 0.0f;
                    float norm_factor = 0.0f;
                    
                    #pragma unroll
                    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
                        if (attention_mask[q_seq * seq_len + k_seq] > 0.0f) {
                            float q_val = query[q_seq * num_heads * head_dim + head * head_dim + dim];
                            float k_val = key[k_seq * num_heads * head_dim + head * head_dim + dim];
                            float score = expf(q_val * k_val / sqrtf(head_dim));
                            norm_factor += score;
                            
                            // Aggregate weighted values
                            for (int v_dim = 0; v_dim < head_dim; v_dim++) {
                                float v_val = value[k_seq * num_heads * head_dim + head * head_dim + v_dim];
                                output_val += score * v_val;
                            }
                        }
                    }
                    
                    if (norm_factor > 0.0f) {
                        output[q_seq * num_heads * head_dim + head * head_dim + dim] = output_val / norm_factor;
                    } else {
                        output[q_seq * num_heads * head_dim + head * head_dim + dim] = 0.0f;
                    }
                }
            }
        }
    }
}

// INTELLECT-3 MoE Expert Routing Kernel
__global__ void intellect3_moe_routing_kernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ expert_gates,
    int* __restrict__ expert_assignments,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < batch_size * seq_len; idx += total_threads) {
        int batch = idx / seq_len;
        int seq = idx % seq_len;
        
        if (batch < batch_size && seq < seq_len) {
            // Top-k expert routing for INTELLECT-3
            float gate_logits[MAX_EXPERTS];
            
            // Compute gate logits for all experts
            #pragma unroll
            for (int expert = 0; expert < MAX_EXPERTS; expert++) {
                float gate_logit = 0.0f;
                
                // Simplified gating computation (would use proper projection in full implementation)
                for (int dim = 0; dim < hidden_dim; dim++) {
                    gate_logit += hidden_states[idx * hidden_dim + dim] * 0.01f; // Simplified weights
                }
                gate_logits[expert] = gate_logit;
            }
            
            // Find top-2 experts (simplified selection)
            int top_experts[ACTIVE_EXPERTS] = {-1, -1};
            float top_scores[ACTIVE_EXPERTS] = {-INFINITY, -INFINITY};
            
            #pragma unroll
            for (int expert = 0; expert < MAX_EXPERTS; expert++) {
                if (gate_logits[expert] > top_scores[0]) {
                    top_scores[1] = top_scores[0];
                    top_experts[1] = top_experts[0];
                    top_scores[0] = gate_logits[expert];
                    top_experts[0] = expert;
                } else if (gate_logits[expert] > top_scores[1]) {
                    top_scores[1] = gate_logits[expert];
                    top_experts[1] = expert;
                }
            }
            
            // Store routing decisions
            expert_assignments[idx * ACTIVE_EXPERTS + 0] = top_experts[0];
            expert_assignments[idx * ACTIVE_EXPERTS + 1] = top_experts[1];
            
            // Store gate values (softmax applied)
            float gate_sum = expf(top_scores[0]) + expf(top_scores[1]);
            expert_gates[idx * ACTIVE_EXPERTS + 0] = expf(top_scores[0]) / gate_sum;
            expert_gates[idx * ACTIVE_EXPERTS + 1] = expf(top_scores[1]) / gate_sum;
        }
    }
}

// MoE Expert Computation Kernel
__global__ void intellect3_expert_forward_kernel(
    const float* __restrict__ input,
    const int* __restrict__ expert_assignments,
    const float* __restrict__ expert_gates,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int expert_id
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < batch_size * seq_len; idx += total_threads) {
        int batch = idx / seq_len;
        int seq = idx % seq_len;
        
        if (batch < batch_size && seq < seq_len) {
            // Check if this expert is active for this token
            bool is_active = false;
            float gate_value = 0.0f;
            
            #pragma unroll
            for (int k = 0; k < ACTIVE_EXPERTS; k++) {
                if (expert_assignments[idx * ACTIVE_EXPERTS + k] == expert_id) {
                    is_active = true;
                    gate_value = expert_gates[idx * ACTIVE_EXPERTS + k];
                    break;
                }
            }
            
            if (is_active) {
                // Apply expert transformation (simplified)
                for (int dim = 0; dim < hidden_dim; dim++) {
                    float input_val = input[idx * hidden_dim + dim];
                    // Simplified expert computation (would use actual weights in full implementation)
                    float expert_out = input_val * 0.5f + 0.1f * sinf(input_val);
                    
                    // Add to output with gating
                    output[idx * hidden_dim + dim] += gate_value * expert_out;
                }
            }
        }
    }
}

// GLM-4 RoPE (Rotary Position Embedding) Kernel
__global__ void glm4_rope_kernel(
    float* __restrict__ queries,
    float* __restrict__ keys,
    const float* __restrict__ position_embeddings,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < batch_size * seq_len * num_heads * head_dim; idx += total_threads) {
        int batch = idx / (seq_len * num_heads * head_dim);
        int temp = idx % (seq_len * num_heads * head_dim);
        int seq = temp / (num_heads * head_dim);
        temp = temp % (num_heads * head_dim);
        int head = temp / head_dim;
        int dim = temp % head_dim;
        
        if (dim < GLM_ROPE_DIM) {
            // GLM-4 RoPE application
            int rope_dim = dim / 2;
            float cos_val = cosf(position_embeddings[seq * GLM_ROPE_DIM + 2 * rope_dim]);
            float sin_val = sinf(position_embeddings[seq * GLM_ROPE_DIM + 2 * rope_dim]);
            
            float* tensor_ptr = (dim < head_dim / 2) ? queries : keys;
            int tensor_idx = ((batch * seq_len + seq) * num_heads + head) * head_dim + dim;
            
            if (dim % 2 == 0) {
                // Even dimensions: cos * x + sin * y
                float x = tensor_ptr[tensor_idx];
                float y = tensor_ptr[tensor_idx + 1];
                tensor_ptr[tensor_idx] = cos_val * x + sin_val * y;
            } else {
                // Odd dimensions: -sin * x + cos * y
                float x = tensor_ptr[tensor_idx - 1];
                float y = tensor_ptr[tensor_idx];
                tensor_ptr[tensor_idx] = -sin_val * x + cos_val * y;
            }
        }
    }
}

// Memory optimization kernels for 106B model
__global__ void optimize_memory_layout_kernel(
    float* __restrict__ weights,
    const int num_params,
    const int quantization_level
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < num_params; idx += total_threads) {
        // Apply memory optimizations for large models
        if (quantization_level == 4) {
            // 4-bit quantization for memory efficiency
            float weight = weights[idx];
            // Simplified quantization (would implement proper quantization in full version)
            int8_t quantized = (int8_t)__float2int_rn(weight * 127.0f);
            weights[idx] = quantized / 127.0f;
        }
    }
}

// Host functions for kernel launching
extern "C" {
    // GLM-4 attention forward pass
    cudaError_t launch_glm4_attention(
        const float* query, const float* key, const float* value,
        const float* attention_mask, float* output,
        int batch_size, int seq_len, int num_heads, int head_dim
    ) {
        dim3 block(256);
        dim3 grid((batch_size * seq_len + block.x - 1) / block.x, num_heads, 1);
        
        glm4_attention_forward_kernel<<<grid, block>>>(
            query, key, value, attention_mask, output,
            batch_size, seq_len, num_heads, head_dim
        );
        
        return cudaGetLastError();
    }
    
    // INTELLECT-3 MoE routing
    cudaError_t launch_intellect3_moe_routing(
        const float* hidden_states, float* expert_gates, int* expert_assignments,
        int batch_size, int seq_len, int hidden_dim
    ) {
        dim3 block(256);
        dim3 grid((batch_size * seq_len + block.x - 1) / block.x, 1, 1);
        
        intellect3_moe_routing_kernel<<<grid, block>>>(
            hidden_states, expert_gates, expert_assignments,
            batch_size, seq_len, hidden_dim
        );
        
        return cudaGetLastError();
    }
    
    // INTELLECT-3 expert computation
    cudaError_t launch_intellect3_expert_forward(
        const float* input, const int* expert_assignments, const float* expert_gates,
        float* output, int batch_size, int seq_len, int hidden_dim, int expert_id
    ) {
        dim3 block(256);
        dim3 grid((batch_size * seq_len + block.x - 1) / block.x, 1, 1);
        
        intellect3_expert_forward_kernel<<<grid, block>>>(
            input, expert_assignments, expert_gates, output,
            batch_size, seq_len, hidden_dim, expert_id
        );
        
        return cudaGetLastError();
    }
    
    // GLM-4 RoPE application
    cudaError_t launch_glm4_rope(
        float* queries, float* keys, const float* position_embeddings,
        int batch_size, int seq_len, int num_heads, int head_dim
    ) {
        dim3 block(256);
        dim3 grid((batch_size * seq_len * num_heads * head_dim + block.x - 1) / block.x, 1, 1);
        
        glm4_rope_kernel<<<grid, block>>>(
            queries, keys, position_embeddings,
            batch_size, seq_len, num_heads, head_dim
        );
        
        return cudaGetLastError();
    }
    
    // Memory optimization for large models
    cudaError_t launch_memory_optimization(
        float* weights, int num_params, int quantization_level
    ) {
        dim3 block(256);
        dim3 grid((num_params + block.x - 1) / block.x, 1, 1);
        
        optimize_memory_layout_kernel<<<grid, block>>>(
            weights, num_params, quantization_level
        );
        
        return cudaGetLastError();
    }
}
