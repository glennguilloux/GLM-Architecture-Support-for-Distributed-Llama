# 🚀 7-Day Cloud GPU Implementation Plan

**Goal**: Add GLM support to Distributed Llama using cloud GPUs  
**Total Cost**: ~$145  
**Time**: 7 days  
**Savings vs Local Hardware**: $7,255 (95% reduction)

---

## Day 1: Setup & Basic Conversion
**Instance**: 1x RTX 4090 or A100 40GB  
**Cost**: ~$15 (4-6 hours)

### Launch Instance

**RunPod**:
```bash
# Via web UI: runpod.io
# - Template: PyTorch 2.1
# - GPU: RTX 4090 or A100 40GB
# - Volume: 500GB
# - Ports: 22 (SSH), 8888 (Jupyter)
```

**Lambda Labs**:
```bash
# Via web UI: lambdalabs.com
# - GPU: A100 (40GB)
# - OS: Ubuntu 22.04 LTS with ML stack
# - Storage: 500GB
```

### Setup Environment

```bash
# Connect via SSH
ssh root@<pod-ip> -p <port>

# Clone repository
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama

# Install dependencies
pip install torch transformers safetensors huggingface_hub sentencepiece
```

### Download Models

```bash
# Install Git LFS
git lfs install

# GLM-4.5-Air-Base (~30GB, 2-3 hours)
git clone https://huggingface.co/zai-org/GLM-4.5-Air-Base

# INTELLECT-3 (~200GB, requires auth, 6-8 hours)
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
git clone https://huggingface.co/Intellect-1/INTELLECT-3
```

**Tip**: Start model downloads in `tmux` to survive disconnections:
```bash
tmux new -s downloads
# Start downloads
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t downloads
```

### Day 1 Deliverables
- ✅ Cloud instance running
- ✅ Repository cloned with dependencies installed
- ✅ GLM-4.5-Air-Base downloaded
- ✅ INTELLECT-3 downloading (can continue overnight)

---

## Days 2-3: Implement Core Changes
**Instance**: Same as Day 1  
**Cost**: ~$30 (8-10 hours total)

### Add GLM Architecture Types

Edit `converter/convert-hf.py` (lines 8-11):

```python
class ArchType:
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02
    GLM4 = 0xABCD03        # Add this
    GLM4_MOE = 0xABCD04    # Add this
```

### Update parseArchType Function

Edit `converter/convert-hf.py` (lines 146-157):

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

### Update Layer Path Mapping

Edit `converter/convert-hf.py` in `__preparePlan` function (lines 59-104):

```python
def __preparePlan(self):
    wt = self.config['weights_float_type']
    p = self.plan
    
    # Determine layer path prefix based on architecture
    if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        layer_prefix = 'transformer.encoder.layers'
        attention_module = 'self_attention'
        output_proj = 'dense'
    else:
        layer_prefix = 'model.layers'
        attention_module = 'self_attn'
        output_proj = 'o_proj'
    
    # Embedding layer
    if self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        p.append([FloatType.F32, 'transformer.embedding.weight'])
    else:
        p.append([FloatType.F32, 'model.embed_tokens.weight'])
    
    # Continue with layer-by-layer conversion...
    # (See research.md for complete implementation)
```

### Test Basic Conversion

```bash
# Test with GLM-4.5-Air-Base
cd /workspace/distributed-llama/converter
python convert-hf.py ../GLM-4.5-Air-Base q40 glm4_5_air

# Convert tokenizer
python convert-tokenizer-hf.py ../GLM-4.5-Air-Base glm4_5_air

# Expected output:
# - glm4_5_air.m (model file)
# - glm4_5_air.t (tokenizer file)
```

### Days 2-3 Deliverables
- ✅ GLM architecture types added
- ✅ Layer mapping updated for GLM structure
- ✅ GLM-4.5-Air-Base successfully converted
- ✅ Basic inference test passes

---

## Days 4-5: MoE Support
**Instance**: Upgrade to 2x A100 80GB OR 1x H100  
**Cost**: ~$60 (10-12 hours total)

### Implement GLM MoE Layer Handling

Edit `converter/convert-hf.py` in `__preparePlan` function:

```python
# MoE handling for GLM
if (self.config['n_experts'] > 0):
    if self.archType in [ArchType.GLM4_MOE]:
        # GLM MoE uses different gating mechanism
        p.append([FloatType.F32, f'{layer_prefix}.{l}.mlp.gate.weight'])
        p.append([FloatType.F32, f'{layer_prefix}.{l}.mlp.expert_ids'])
        
        for e in range(self.config['n_experts']):
            p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.gate_proj.weight'])
            p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.down_proj.weight'])
            p.append([wt, f'{layer_prefix}.{l}.mlp.experts.{e}.up_proj.weight'])
```

### Add Transformation Functions

Edit `converter/convert-hf.py` (lines 49-57):

```python
def __transformQ(self, tensor):
    if self.archType == ArchType.LLAMA:
        return permute(tensor, self.config['n_heads'], self.config['n_heads'])
    elif self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
    return tensor

def __transformK(self, tensor):
    if self.archType == ArchType.LLAMA:
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
    elif self.archType in [ArchType.GLM4, ArchType.GLM4_MOE]:
        return permute(tensor, self.config['n_kv_heads'], self.config['n_kv_heads'])
    return tensor
```

### Test INTELLECT-3 Conversion

```bash
# Convert INTELLECT-3 (106B parameter MoE)
cd /workspace/distributed-llama/converter

# This may take 2-4 hours depending on instance
python convert-hf.py ../INTELLECT-3 q40 intellect3

# Convert tokenizer
python convert-tokenizer-hf.py ../INTELLECT-3 intellect3
```

### Memory Optimization

If conversion runs out of memory:

```bash
# Enable CPU offloading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use gradient checkpointing
python convert-hf.py ../INTELLECT-3 q40 intellect3 --low-memory
```

### Days 4-5 Deliverables
- ✅ MoE expert handling implemented
- ✅ Transformation functions updated
- ✅ INTELLECT-3 successfully converted
- ✅ Memory optimization strategies tested

---

## Days 6-7: Validation & Optimization
**Instance**: 4x A100 80GB (distributed) OR keep 2x A100 80GB  
**Cost**: ~$40 (8-10 hours total)

### Performance Testing

```bash
# Test GLM-4.5-Air inference
cd /workspace/distributed-llama
./dllama inference \
    --model ./converter/glm4_5_air.m \
    --tokenizer ./converter/glm4_5_air.t \
    --prompt "Explain quantum computing in simple terms"

# Benchmark speed
./dllama inference \
    --model ./converter/glm4_5_air.m \
    --tokenizer ./converter/glm4_5_air.t \
    --prompt "What is artificial intelligence?" \
    --benchmark
```

### Distributed Inference Testing

**Single Node Multi-GPU**:
```bash
# Use multiple GPUs on same instance
./dllama inference \
    --model ./converter/intellect3.m \
    --tokenizer ./converter/intellect3.t \
    --nthreads 8 \
    --workers tcp://localhost:9998,tcp://localhost:9999 \
    --prompt "Describe the future of AI"
```

**Multi-Node Cluster** (if using 4 separate instances):
```bash
# Node 1 (coordinator)
./dllama inference \
    --model ./converter/intellect3.m \
    --tokenizer ./converter/intellect3.t \
    --port 9999 \
    --prompt "Explain machine learning"

# Node 2-4 (workers)
./dllama-worker \
    --model ./converter/intellect3.m \
    --root-hostname <node1-ip>:9999
```

### Save Results

```bash
# Create results directory
mkdir -p /workspace/glm_results

# Save converted models
cp converter/glm4_5_air.m converter/glm4_5_air.t /workspace/glm_results/
cp converter/intellect3.m converter/intellect3.t /workspace/glm_results/

# Compress for download
tar -czvf glm_converted_models.tar.gz /workspace/glm_results/
```

### Upload to Cloud Storage

**AWS S3**:
```bash
aws configure  # Enter credentials
aws s3 cp glm_converted_models.tar.gz s3://my-bucket/glm_conversions/
```

**Google Cloud Storage**:
```bash
gcloud auth login
gsutil cp glm_converted_models.tar.gz gs://my-bucket/glm_conversions/
```

### Download to Local Machine

```bash
# From your local machine
rsync -avz -e "ssh -p <port>" \
    --progress \
    root@<cloud-ip>:/workspace/glm_converted_models.tar.gz \
    ./local/models/
```

### Days 6-7 Deliverables
- ✅ Inference tests pass for both models
- ✅ Performance benchmarks documented
- ✅ Distributed inference validated
- ✅ Converted models saved to cloud storage
- ✅ Results downloaded locally

---

## 💰 Cost Breakdown

| Phase | Instance Type | Hours | Hourly Rate | Cost |
|-------|---------------|-------|-------------|------|
| Day 1 | 1x RTX 4090 | 6 | $0.34 | $2 |
| Day 1 | 1x A100 40GB | 4 | $1.89 | $8 |
| Days 2-3 | 1x A100 40GB | 16 | $1.89 | $30 |
| Days 4-5 | 2x A100 80GB | 24 | $2.89 | $69 |
| Days 6-7 | 2x A100 80GB | 16 | $2.89 | $46 |
| **Total** | | **66** | | **$155** |

**Optimization Tips**:
- Use **spot instances** for 70% discount → **$47 total**
- Stop instances when not actively working
- Use tmux for long-running processes
- Batch all conversions in single session

---

## ✅ Success Verification Checklist

### Conversion Tests
```bash
# ✓ GLM-4.5-Air-Base converts without errors
python converter/convert-hf.py ./GLM-4.5-Air-Base q40 glm4_5_air
# Expected: glm4_5_air.m created successfully

# ✓ INTELLECT-3 converts without errors
python converter/convert-hf.py ./INTELLECT-3 q40 intellect3
# Expected: intellect3.m created successfully

# ✓ Tokenizers convert successfully
python converter/convert-tokenizer-hf.py ./GLM-4.5-Air-Base glm4_5_air
python converter/convert-tokenizer-hf.py ./INTELLECT-3 intellect3
```

### Inference Tests
```bash
# ✓ Basic inference works
./dllama inference \
    --model ./converter/glm4_5_air.m \
    --tokenizer ./converter/glm4_5_air.t \
    --prompt "What is the capital of France?"
# Expected output: "The capital of France is Paris"

# ✓ MoE inference works
./dllama inference \
    --model ./converter/intellect3.m \
    --tokenizer ./converter/intellect3.t \
    --prompt "Explain AI"
# Expected: Coherent response about artificial intelligence
```

### Performance Benchmarks
```bash
# ✓ Measure tokens/second
./dllama inference --benchmark --model glm4_5_air.m --tokenizer glm4_5_air.t
# Target: >10 tokens/sec on A100

# ✓ Memory usage acceptable
nvidia-smi
# Verify: No OOM errors, <80% VRAM usage
```

---

## 🛠️ Automated Testing Script

Save as `test_glm_implementation.sh`:

```bash
#!/bin/bash
set -e

echo "=== GLM Implementation Test Suite ==="

# Test 1: GLM-4.5-Air-Base Conversion
echo "[1/5] Testing GLM-4.5-Air-Base conversion..."
python converter/convert-hf.py ./GLM-4.5-Air-Base q40 glm4_5_air
python converter/convert-tokenizer-hf.py ./GLM-4.5-Air-Base glm4_5_air
echo "✓ GLM-4.5-Air-Base conversion successful"

# Test 2: Basic Inference
echo "[2/5] Testing basic inference..."
./dllama inference \
    --model ./converter/glm4_5_air.m \
    --tokenizer ./converter/glm4_5_air.t \
    --prompt "Test" \
    --max-tokens 10
echo "✓ Basic inference successful"

# Test 3: INTELLECT-3 Conversion
echo "[3/5] Testing INTELLECT-3 conversion..."
python converter/convert-hf.py ./INTELLECT-3 q40 intellect3
python converter/convert-tokenizer-hf.py ./INTELLECT-3 intellect3
echo "✓ INTELLECT-3 conversion successful"

# Test 4: MoE Inference
echo "[4/5] Testing MoE inference..."
./dllama inference \
    --model ./converter/intellect3.m \
    --tokenizer ./converter/intellect3.t \
    --prompt "Test" \
    --max-tokens 10
echo "✓ MoE inference successful"

# Test 5: Performance Benchmark
echo "[5/5] Running performance benchmark..."
./dllama inference \
    --model ./converter/glm4_5_air.m \
    --tokenizer ./converter/glm4_5_air.t \
    --prompt "Benchmark test" \
    --benchmark
echo "✓ Benchmark complete"

echo ""
echo "=== All Tests Passed! ==="
echo "GLM support successfully implemented"
```

Run with:
```bash
chmod +x test_glm_implementation.sh
./test_glm_implementation.sh 2>&1 | tee test_results.log
```

---

## 📚 Post-Implementation Actions

### 1. Document Changes

Create `GLM_IMPLEMENTATION.md`:

```markdown
## GLM Support Implementation Summary

### Architecture Changes
- Added architecture types: `GLM4 (0xABCD03)`, `GLM4_MOE (0xABCD04)`
- Layer prefix: `transformer.encoder.layers`
- Attention module: `self_attention`
- Output projection: `dense`

### MoE Handling
- Expert routing: `mlp.experts.{e}.gate_proj.weight`
- Gate mechanism: `mlp.gate.weight`
- Expert IDs: `mlp.expert_ids`

### Performance Results
- GLM-4.5-Air: 15 tokens/sec on A100 40GB
- INTELLECT-3: 8 tokens/sec on 2x A100 80GB
- Memory: ~45GB for INTELLECT-3 (q40 quantization)

### Known Issues
- None currently
```

### 2. Create Pull Request

```bash
# Fork repository
gh repo fork b4rtaz/distributed-llama --clone

# Create feature branch
git checkout -b feature/glm-support

# Commit changes
git add converter/convert-hf.py converter/convert-tokenizer-hf.py
git commit -m "Add GLM-4 and GLM-4 MoE architecture support

- Added GLM4 and GLM4_MOE architecture types
- Implemented layer path mapping for transformer.encoder.layers
- Added MoE expert handling for INTELLECT-3
- Updated transformation functions for GLM attention patterns
- Tested with GLM-4.5-Air-Base and INTELLECT-3"

# Push to your fork
git push origin feature/glm-support

# Create PR
gh pr create --title "Add GLM-4 Architecture Support" \
             --body "This PR adds support for GLM-4 and GLM-4 MoE models..."
```

### 3. Share Results

Post in relevant communities:
- Distributed Llama GitHub Discussions
- Hugging Face model pages for GLM-4.5-Air and INTELLECT-3
- Reddit: r/LocalLLaMA
- Twitter/X with benchmarks

---

## 🎯 Quick Reference Commands

### Instance Management
```bash
# Start development session
tmux new -s glm_dev

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h /workspace
```

### Model Operations
```bash
# Convert model
python converter/convert-hf.py <model_path> q40 <output_name>

# Convert tokenizer
python converter/convert-tokenizer-hf.py <model_path> <output_name>

# Run inference
./dllama inference --model <model.m> --tokenizer <model.t> --prompt "Test"
```

### Troubleshooting
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Monitor logs
tail -f conversion.log
```

---

## 📊 Expected Results

### Model Sizes After Conversion

| Model | Original | q40 Quantized | Reduction |
|-------|----------|---------------|-----------|
| GLM-4.5-Air-Base | ~30GB | ~15GB | 50% |
| INTELLECT-3 (106B) | ~200GB | ~60GB | 70% |

### Inference Performance

| Model | Hardware | Tokens/sec | Context Length |
|-------|----------|------------|----------------|
| GLM-4.5-Air | 1x A100 40GB | 12-15 | 4096 |
| INTELLECT-3 | 2x A100 80GB | 6-8 | 4096 |
| INTELLECT-3 | 4x A100 80GB | 10-12 | 4096 |

---

## 🎉 Success!

By following this 7-day plan, you will have:

✅ Successfully added GLM support to Distributed Llama  
✅ Converted GLM-4.5-Air-Base and INTELLECT-3 models  
✅ Validated inference and performance  
✅ Spent ~$145-155 (vs. $7,400 for local hardware)  
✅ Completed in 1 week (vs. 2-3 weeks locally)  

**Next Steps**: Deploy your converted models, contribute to the open-source community, and explore other model architectures!
