# Research: Adding GLM Support to Distributed Llama Converter

## Executive Summary

To add support for GLM models (specifically GLM-4.5-Air-Base, which INTELLECT-3 is based on) to the Distributed Llama converter, significant modifications are required across multiple files. The current converter supports LLaMA, Mistral, and Qwen architectures, but lacks GLM-specific layer mapping, transformation functions, and configuration handling.

## Current Converter Architecture

### Supported Architectures (from `convert-hf.py`)

```python
class ArchType:
    LLAMA = 0xABCD00        # LLaMA, Mistral
    QWEN3 = 0xABCD01        # Qwen2, Qwen2.5, Qwen3
    QWEN3_MOE = 0xABCD02    # Qwen3 MoE
```

### Key Architecture Differences: GLM vs Supported Models

| Feature | LLaMA/Qwen | GLM-4.5-Air |
|---------|------------|-------------|
| **Model Type** | `llama`, `mistral`, `qwen3` | `glm4` or `glm4_moe` |
| **Layer Path** | `model.layers.{l}.*` | `transformer.encoder.layers.{l}.*` |
| **Attention Module** | `self_attn` | `self_attention` |
| **Output Projection** | `o_proj.weight` | `dense.weight` |
| **QKV Projection** | Separate `q_proj`, `k_proj`, `v_proj` | Combined `query_key_value` or separate |
| **MLP Structure** | Standard FFN | GLU variant with `dense_4h_to_h` |
| **Norm Layers** | RMSNorm | Different structure |
| **MoE Support** | Via Qwen3_MOE | Native (12B active/106B total) |

## Required Changes

### 1. Add GLM Architecture Type (`convert-hf.py`)

**Location**: Lines 8-11

```python
class ArchType:
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02
    GLM4 = 0xABCD03        # Add GLM-4 series
    GLM4_MOE = 0xABCD04    # Add GLM-4 MoE
```

### 2. Update parseArchType Function

**Location**: Lines 146-157

Add GLM model types:

```python
def parseArchType(type: str):
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
        'qwen2': ArchType.QWEN3,
        'qwen2.5': ArchType.QWEN3,
        'qwen3': ArchType.QWEN3,
        'qwen3_moe': ArchType.QWEN3_MOE,
        'glm4': ArchType.GLM4,              # Add
        'glm4_moe': ArchType.GLM4_MOE,      # Add
    }.get(type)
    if (archType is None):
        raise Exception(f'Unsupported arch type: {type}')
    return archType
```

### 3. Add GLM-Specific Transformation Functions

**Location**: Lines 49-57

GLM uses multi-query attention with different head arrangements:

```python
def __transformQ(self, tensor):
    if self.archType == ArchType.LLAMA:
        return permute(tensor, self.config['n_heads'], self.config['n_heads'])
    elif self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        # GLM uses multi-query attention with different head arrangement
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
    return tensor

def __transformK(self, tensor):
    if self.archType == ArchType.LLAMA:
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
    elif self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        # GLM uses a different KV cache structure
        return permute(tensor, self.config['n_kv_heads'], self.config['n_kv_heads'])
    return tensor
```

### 4. Modify Layer Path Mapping (`__preparePlan` function)

**Location**: Lines 59-104

GLM uses different layer paths:

```python
def __preparePlan(self):
    wt = self.config['weights_float_type']
    p = self.plan
    
    # Determine layer path prefix based on architecture
    if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        layer_prefix = 'transformer.encoder.layers'
        attention_module = 'self_attention'
        mlp_module = 'mlp'
        output_proj = 'dense'
    else:
        layer_prefix = 'model.layers'
        attention_module = 'self_attn'
        mlp_module = 'mlp'
        output_proj = 'o_proj'
    
    p.append([FloatType.F32, 'transformer.embedding' if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE] else 'model.embed_tokens.weight'])
    
    for l in range(0, self.config['n_layers']):
        # Attention projections with GLM-specific paths
        p.append([wt, self.__transformQ,
            f'{layer_prefix}.{l}.{attention_module}.q_proj.weight' if self.archType not in [ArchType.GLM4, ArchType.GLM4_MOE] else f'{layer_prefix}.{l}.{attention_module}.query.weight'])
        p.append([wt, self.__transformK,
            f'{layer_prefix}.{l}.{attention_module}.k_proj.weight' if self.archType not in [ArchType.GLM4, ArchType.GLM4_MOE] else f'{layer_prefix}.{l}.{attention_module}.key.weight'])
        p.append([wt,
            f'{layer_prefix}.{l}.{attention_module}.v_proj.weight' if self.archType not in [ArchType.GLM4, ArchType.GLM4_MOE] else f'{layer_prefix}.{l}.{attention_module}.value.weight'])
        p.append([wt,
            f'{layer_prefix}.{l}.{attention_module}.{output_proj}.weight'])
        
        # MoE handling for GLM
        if (self.config['n_experts'] > 0):
            if self.archType in [ArchType.GLM4_MOE]:
                # GLM MoE uses a different gating mechanism
                p.append([FloatType.F32, f'{layer_prefix}.{l}.{mlp_module}.gate.weight'])
                # GLM MoE has specific expert routing parameters
                p.append([FloatType.F32, f'{layer_prefix}.{l}.{mlp_module}.expert_ids'])
                for e in range(self.config['n_experts']):
                    p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.experts.{e}.gate_proj.weight'])
                    p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.experts.{e}.down_proj.weight'])
                    p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.experts.{e}.up_proj.weight'])
            else:
                p.append([FloatType.F32, f'{layer_prefix}.{l}.mlp.gate.weight'])
                for e in range(self.config['n_experts']):
                    p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.gate_proj.weight'])
                    p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.down_proj.weight'])
                    p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.up_proj.weight'])
        else:
            # Standard MLP with GLM-specific layer names
            p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.gate_proj.weight'])
            p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.{"dense_4h_to_h" if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE] else "down_proj"}.weight'])
            p.append([wt, f'{layer_prefix}.{l}.{mlp_module}.up_proj.weight'])
        
        # Norm layers (GLM has different norm structure)
        if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
            p.append([FloatType.F32, f'{layer_prefix}.{l}.attention.layernorm.weight'])
            p.append([FloatType.F32, f'{layer_prefix}.{l}.ffn.layernorm.weight'])
        else:
            if (self.archType == ArchType.QWEN3 or self.archType == ArchType.QWEN3_MOE):
                p.append([FloatType.F32, f'{layer_prefix}.{l}.self_attn.q_norm.weight'])
                p.append([FloatType.F32, f'{layer_prefix}.{l}.self_attn.k_norm.weight'])
            p.append([FloatType.F32, f'{layer_prefix}.{l}.input_layernorm.weight'])
            p.append([FloatType.F32, f'{layer_prefix}.{l}.post_attention_layernorm.weight'])
    
    # Final norm and output layer
    if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        p.append([FloatType.F32, 'transformer.encoder.final_layernorm.weight'])
        p.append([wt, 'transformer.output_layer.weight', 'transformer.embedding.weight'])
    else:
        p.append([FloatType.F32, 'model.norm.weight'])
        p.append([wt, 'lm_head.weight', 'model.embed_tokens.weight'])
```

### 5. Handle GLM-Specific Configuration

**Location**: Lines 183-238

GLM models have different config parameters:

```python
def loadConfig(folderPath: str, weightsFloatType: int):
    # ... existing code ...
    
    result = {
        'version': 0,
        'arch_type': parseArchType(config['model_type']),
        'hidden_act': parseHiddenAct(config['hidden_act']),  # May need GLM-specific parser
        'dim': config['hidden_size'],
        'hidden_dim': config['intermediate_size'],
        'n_layers': config['num_hidden_layers'],
        'n_heads': config['num_attention_heads'],
        'n_kv_heads': config.get('num_key_value_heads', config['num_attention_heads']),
        'weights_float_type': weightsFloatType,
        'max_seq_len': config['max_position_embeddings'],
        'vocab_size': config['vocab_size'],
        'files': files,
    }
    
    # GLM-specific configurations
    if config['model_type'] in ['glm4', 'glm4_moe']:
        # Handle GLM-specific parameters
        result['moe_hidden_dim'] = config.get('moe_intermediate_size', result['hidden_dim'])
        result['norm_epsilon'] = parseRmsNormEpsilon(config.get('rms_norm_eps', 1e-5))
        
        # GLM rope configuration (different from LLaMA)
        if 'rope_theta' in config:
            result['rope_theta'] = config['rope_theta']
        
        # GLM uses a different positional encoding approach
        result['position_encoding_2d'] = config.get('position_encoding_2d', True)
        result['use_cache'] = config.get('use_cache', True)
        
        # Memory optimization for MoE models
        if self.archType == ArchType.GLM4_MOE:
            # Only load active experts to save memory
            result['load_experts_on_demand'] = True
            result['max_active_experts'] = config.get('num_experts_per_tok', 2)
    
    # ... rest of existing config handling ...
```

### 6. Update Tokenizer Converter

**Location**: `convert-tokenizer-hf.py`

GLM models use different tokenizer configurations:

```python
def resolve(self):
    cls = self.tokenizerConfig['tokenizer_class']
    if (cls == 'PreTrainedTokenizerFast' or
        cls == 'LlamaTokenizerFast' or
        cls == 'Qwen2Tokenizer' or
        cls == 'ChatGLMTokenizer'):  # Add GLM tokenizer
        return self.resolvePreTrainedTokenizerFast()
    if (cls == 'LlamaTokenizer' or
        cls == 'ChatGLMTokenizer'):  # Add GLM tokenizer
        return self.resolveLlamaTokenizer()
    raise Exception(f'Tokenizer {cls} is not supported')
```

### 7. Add to Launch Script

**Location**: `launch.py`

```python
MODELS = {
    # ... existing models ...
    'intellect3_q40': [
        # Split model across multiple machines for 106B parameter MoE
        ['path/to/converted/model/part1.m', 'path/to/converted/model/part2.m'],
        'path/to/converted/tokenizer/file.t',
        'q40', 'q80', 'chat', '--max-seq-len 4096', '--distributed-inference'
    ],
}
```

**For large MoE models like INTELLECT-3, consider optimizing for distributed inference**.

## Testing Strategy

### 1. Verify GLM Model Structure
```bash
# Download and inspect GLM-4.5-Air-Base structure
git clone https://huggingface.co/zai-org/GLM-4.5-Air-Base
ls GLM-4.5-Air-Base/
# Check config.json, tokenizer files, and safetensors
```

### 2. Test Converter Extensions
```bash
# Test basic GLM model conversion
cd converter
python convert-hf.py /path/to/GLM-4.5-Air-Base q40 glm4_5_air
python convert-tokenizer-hf.py /path/to/GLM-4.5-Air-Base glm4_5_air
```

### 3. Test with INTELLECT-3
```bash
# After converter is updated, test with INTELLECT-3
python convert-hf.py /path/to/INTELLECT-3 q40 intellect3
python convert-tokenizer-hf.py /path/to/INTELLECT-3 intellect3
```

### 4. Enhanced Testing Strategy
- **Unit Testing**: Create unit tests for each transformation function
- **Integration Testing**: Test with multiple GLM model sizes
- **Performance Testing**: Measure inference speed and memory usage

## Key Implementation Challenges

### 1. **Layer Name Mapping**
GLM uses completely different layer paths (`transformer.encoder.layers` vs `model.layers`) and different attention module names.

### 2. **Attention Pattern Differences**
GLM may require different Q/K tensor permutations compared to LLaMA models.

### 3. **MoE Architecture**
INTELLECT-3 is a 106B parameter MoE with 12B active parameters, requiring proper expert routing support.

### 4. **Configuration Parameters**
GLM models have different configuration schema and may require new parameter parsing.

### 5. **Norm Layer Structure**
GLM uses different normalization layer structures and naming conventions.

### 6. **Tokenizer Compatibility**
GLM tokenizers may have different special token handling and chat template structures.

## Development Phases

### Phase 1: Basic GLM Support
1. Add GLM architecture types
2. Implement basic layer path mapping
3. Test with simple GLM-4.5-Air-Base conversion

### Phase 2: MoE Support
1. Implement GLM-specific MoE layer handling
2. Test with larger GLM models
3. Optimize memory usage for MoE models

### Phase 3: INTELLECT-3 Integration
1. Test conversion with actual INTELLECT-3 model
2. Validate inference performance
3. Add to launch script for easy access

## Estimated Development Time

- **Phase 1**: 3-4 days (basic GLM support with attention mechanism differences)
- **Phase 2**: 4-5 days (MoE support with memory optimization)
- **Phase 3**: 2-3 days (testing and integration)
- **Total**: 2-3 weeks for full implementation

## Success Criteria

1. Successfully convert GLM-4.5-Air-Base to Distributed Llama format
2. Successfully convert INTELLECT-3 model
3. Models load and run inference without errors
4. Performance is acceptable for distributed setup
5. Integration with launch.py works seamlessly

## Additional Resources Needed

1. **GLM Model Access**: Access to GLM-4.5-Air-Base and INTELLECT-3 models
2. **Testing Infrastructure**: Multiple machines for distributed testing
3. **Documentation**: GLM-specific configuration parameters
4. **Community Support**: Possibly reach out to GLM developers for architecture details

## Documentation Recommendations

After implementation, update the documentation to include:

- GLM-specific configuration options
- Performance expectations for MoE models
- Memory requirements for different model sizes

This research provides a comprehensive roadmap for adding GLM support to Distributed Llama, enabling conversion and usage of INTELLECT-3 and other GLM-based models.