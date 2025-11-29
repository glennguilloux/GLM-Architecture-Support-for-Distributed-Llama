/*
 * CUDA Attention Kernels for GLM Models
 * Optimized attention mechanisms for GLM-4 and INTELLECT-3
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized attention kernel with shared memory usage
__global__ void optimized_glm4_attention_kernel(
    const half* __restrict__ queries,
    const half* __restrict__ keys,
    const half* __restrict__ values,
    const float* __restrict__ attention_mask,
    half* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    extern __shared__ float shared_mem[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    
    if (head >= num_heads || batch >= batch_size) return;
    
    const int head_offset = batch * seq_len * num_heads * head_dim + head * head_dim;
    const int q_offset = head_offset;
    const int k_offset = head_offset;
    const int v_offset = head_offset;
    
    // Shared memory layout: [queries_block, keys_block, attention_scores_block]
    const int queries_start = 0;
    const int keys_start = blockDim.x * head_dim;
    const int scores_start = keys_start + blockDim.x * head_dim;
    
    // Load query block into shared memory
    if (threadIdx.x < seq_len) {
        for (int dim = 0; dim < head_dim; dim += 4) {
            float4 q_vec = reinterpret_cast<const float4*>(
                &queries[q_offset + threadIdx.x * head_dim + dim]
            )[0];
            
            float4* q_shared = reinterpret_cast<float4*>(
                &shared_mem[queries_start + threadIdx.x * head_dim + dim]
            );
            q_shared[0] = q_vec;
        }
    }
    
    __syncthreads();
    
    // Compute attention scores
    float* scores = &shared_mem[scores_start];
    
    for (int k_seq = threadIdx.x; k_seq < seq_len; k_seq += blockDim.x) {
        float score = 0.0f;
        
        // Compute dot product for attention score
        #pragma unroll
        for (int dim = 0; dim < head_dim; dim++) {
            float q_val = shared_mem[queries_start + threadIdx.x * head_dim + dim];
            float k_val = shared_mem[keys_start + k_seq * head_dim + dim];
            score += q_val * k_val;
        }
        
        // Apply attention mask
        if (attention_mask[batch * seq_len * seq_len + threadIdx.x * seq_len + k_seq] > 0.0f) {
            scores[threadIdx.x * seq_len + k_seq] = score / sqrtf(head_dim);
        } else {
            scores[threadIdx.x * seq_len + k_seq] = -INFINITY;
        }
    }
    
    __syncthreads();
    
    // Apply softmax
    float max_score = -INFINITY;
    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
        max_score = fmaxf(max_score, scores[threadIdx.x * seq_len + k_seq]);
    }
    
    float sum_exp = 0.0f;
    for (int k_seq = 0; k_seq < seq_len; k_seq++) {
        if (scores[threadIdx.x * seq_len + k_seq] > -INFINITY) {
            scores[threadIdx.x * seq_len + k_seq] = expf(scores[threadIdx.x * seq_len + k_seq] - max_score);
            sum_exp += scores[threadIdx.x * seq_len + k_seq];
        }
    }
    
    // Normalize and compute output
    half* out_ptr = &output[q_offset + threadIdx.x * head_dim];
    
    for (int dim = 0; dim < head_dim; dim++) {
        float out_val = 0.0f;
        
        for (int k_seq = 0; k_seq < seq_len; k_seq++) {
            if (sum_exp > 0.0f && scores[threadIdx.x * seq_len + k_seq] > -INFINITY) {
                float weight = scores[threadIdx.x * seq_len + k_seq] / sum_exp;
                float v_val = __half2float(values[v_offset + k_seq * head_dim + dim]);
                out_val += weight * v_val;
            }
        }
        
        out_ptr[dim] = __float2half(out_val);
    }
}
