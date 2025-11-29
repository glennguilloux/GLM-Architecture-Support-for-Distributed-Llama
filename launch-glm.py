#!/usr/bin/env python3
"""
Extended Launcher for GLM Architecture Support in Distributed Llama

This script extends the original distributed-llama launcher to support:
- GLM-4 models
- INTELLECT-3 (106B MoE) models
- Consumer hardware optimization
- Distributed inference setup

Usage:
    python launch-glm.py glm_4_9b_instruct_q40
    python launch-glm.py intellect3_106b_moe_q40 --nodes 4
    python launch-glm.py chat --model glm_4_9b_instruct_q40
    python launch-glm.py worker --model intellect3_106b_moe_q40
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Model configurations
GLM_MODELS = {
    "glm_4_9b_instruct_q40": {
        "size": "9B",
        "memory_gb": 7,
        "quantization": "q40",
        "architecture": "glm_4",
        "download_url": "https://huggingface.co/THUDM/glm-4-9b-chat/resolve/main/glm-4-9b-chat-q4_0.gguf"
    },
    "glm_4_4b_instruct_q40": {
        "size": "4B", 
        "memory_gb": 3,
        "quantization": "q40",
        "architecture": "glm_4",
        "download_url": "https://huggingface.co/THUDM/glm-4-4b/resolve/main/glm-4-4b-q4_0.gguf"
    },
    "intellect3_106b_moe_q40": {
        "size": "106B",
        "memory_gb": 13,  # With aggressive quantization
        "quantization": "q40",
        "architecture": "intellect_3",
        "moe_experts": 16,
        "active_experts": 2,
        "download_url": "https://huggingface.co/intellect-ai/intellect-3-106b/resolve/main/intellect3-106b-moeq40.gguf"
    }
}

class GLMLauncher:
    def __init__(self):
        self.models_dir = Path.home() / ".cache" / "glm-distributed"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_hardware(self) -> Dict[str, any]:
        """Detect available hardware for optimization"""
        import psutil
        
        # GPU detection
        gpu_info = {"available": False, "memory_gb": 0, "count": 0}
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info["available"] = True
                gpu_info["count"] = len(gpus)
                gpu_info["memory_gb"] = gpus[0].memoryTotal // 1024  # Convert to GB
        except ImportError:
            print("GPUtil not available, GPU detection skipped")
        
        # System memory
        system_memory_gb = psutil.virtual_memory().total // (1024**3)
        
        return {
            "gpu": gpu_info,
            "system_memory_gb": system_memory_gb,
            "cpu_count": psutil.cpu_count()
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get local path for model, download if needed"""
        model_config = GLM_MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.models_dir / f"{model_name}.gguf"
        
        if not model_path.exists():
            print(f"Model {model_name} not found locally.")
            print(f"Downloading from: {model_config['download_url']}")
            
            # Download model
            subprocess.run([
                "wget", "-O", str(model_path), model_config['download_url']
            ], check=True)
            
            print(f"Model downloaded to: {model_path}")
        
        return model_path
    
    def optimize_for_hardware(self, model_name: str, hardware: Dict) -> List[str]:
        """Generate optimized flags for available hardware"""
        model_config = GLM_MODELS[model_name]
        flags = []
        
        # Memory optimization
        gpu_memory = hardware["gpu"]["memory_gb"]
        model_memory = model_config["memory_gb"]
        
        if gpu_memory < model_memory:
            print(f"Warning: Model requires {model_memory}GB, but GPU has {gpu_memory}GB")
            flags.extend([
                "--enable-quantization",
                "--quantization-level", "q4",
                "--enable-cpu-offloading",
                "--cpu-offload-threshold", "0.3"
            ])
        
        # MoE specific optimizations
        if model_config.get("architecture") == "intellect_3":
            flags.extend([
                "--moe-mode", "distributed",
                "--max-experts-cached", "3",
                "--enable-expert-swapping",
                "--expert-cache-size", "2GB"
            ])
        
        # GPU optimization
        if hardware["gpu"]["available"]:
            if hardware["gpu"]["count"] > 1:
                flags.extend([
                    "--multi-gpu",
                    "--gpu-count", str(hardware["gpu"]["count"])
                ])
            
            flags.extend([
                "--gpu-memory-pool", str(gpu_memory),
                "--enable-mixed-precision"
            ])
        
        return flags
    
    def run_inference(self, model_name: str, prompt: str = None, **kwargs):
        """Run inference with GLM model"""
        hardware = self.detect_hardware()
        model_path = self.get_model_path(model_name)
        model_config = GLM_MODELS[model_name]
        
        # Build command
        cmd = ["./dllama", "inference"]
        cmd.extend(["--model", str(model_path)])
        cmd.extend(["--architecture", model_config["architecture"]])
        
        # Add optimization flags
        optimization_flags = self.optimize_for_hardware(model_name, hardware)
        cmd.extend(optimization_flags)
        
        # Add inference flags
        if prompt:
            cmd.extend(["--prompt", prompt])
        if kwargs.get("max_tokens"):
            cmd.extend(["--max-tokens", str(kwargs["max_tokens"])])
        if kwargs.get("temperature"):
            cmd.extend(["--temperature", str(kwargs["temperature"])])
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    def run_chat(self, model_name: str, **kwargs):
        """Run interactive chat with GLM model"""
        hardware = self.detect_hardware()
        model_path = self.get_model_path(model_name)
        model_config = GLM_MODELS[model_name]
        
        cmd = ["./dllama", "chat"]
        cmd.extend(["--model", str(model_path)])
        cmd.extend(["--architecture", model_config["architecture"]])
        
        # Add optimization flags
        optimization_flags = self.optimize_for_hardware(model_name, hardware)
        cmd.extend(optimization_flags)
        
        print(f"Starting chat: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    def run_worker(self, model_name: str, nodes: int = 1, **kwargs):
        """Run worker node for distributed inference"""
        hardware = self.detect_hardware()
        model_path = self.get_model_path(model_name)
        model_config = GLM_MODELS[model_name]
        
        cmd = ["./dllama", "worker"]
        cmd.extend(["--model", str(model_path)])
        cmd.extend(["--architecture", model_config["architecture"]])
        
        # Distributed inference setup
        if nodes > 1:
            cmd.extend([
                "--distributed",
                "--nodes", str(nodes),
                "--node-id", "0"  # Will be set by orchestration script
            ])
        
        # Add optimization flags
        optimization_flags = self.optimize_for_hardware(model_name, hardware)
        cmd.extend(optimization_flags)
        
        print(f"Starting worker: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    def setup_distributed_cluster(self, model_name: str, node_ips: List[str]):
        """Set up distributed inference cluster"""
        print(f"Setting up distributed cluster for {model_name}")
        print(f"Node IPs: {node_ips}")
        
        # Create cluster configuration
        cluster_config = {
            "model": model_name,
            "nodes": node_ips,
            "architecture": GLM_MODELS[model_name]["architecture"]
        }
        
        config_path = self.models_dir / "cluster_config.json"
        with open(config_path, 'w') as f:
            json.dump(cluster_config, f, indent=2)
        
        print(f"Cluster configuration saved to: {config_path}")
        
        # Generate startup scripts for each node
        for i, ip in enumerate(node_ips):
            script_path = self.models_dir / f"start_node_{i}.sh"
            with open(script_path, 'w') as f:
                f.write(f"""#!/bin/bash
# Startup script for node {i} ({ip})

export NODE_ID={i}
export CLUSTER_CONFIG={config_path}

python {__file__} worker --model {model_name} --nodes {len(node_ips)} --node-id {i}
""")
            script_path.chmod(0o755)
            print(f"Generated startup script: {script_path}")
    
    def list_models(self):
        """List available GLM models"""
        print("Available GLM Models:")
        print("-" * 50)
        
        for name, config in GLM_MODELS.items():
            print(f"Model: {name}")
            print(f"  Size: {config['size']}")
            print(f"  Architecture: {config['architecture']}")
            print(f"  Memory Required: {config['memory_gb']}GB")
            if config['architecture'] == 'intellect_3':
                print(f"  MoE Experts: {config['moe_experts']}")
                print(f"  Active Experts: {config['active_experts']}")
            print(f"  Download URL: {config['download_url']}")
            print()
    
    def benchmark_model(self, model_name: str):
        """Run performance benchmark on GLM model"""
        hardware = self.detect_hardware()
        model_path = self.get_model_path(model_name)
        model_config = GLM_MODELS[model_name]
        
        print(f"Benchmarking {model_name}...")
        print(f"Hardware detected: {json.dumps(hardware, indent=2)}")
        
        # Benchmark command
        cmd = ["./dllama", "benchmark"]
        cmd.extend(["--model", str(model_path)])
        cmd.extend(["--architecture", model_config["architecture"]])
        cmd.extend(["--test-prompts", "10"])
        
        optimization_flags = self.optimize_for_hardware(model_name, hardware)
        cmd.extend(optimization_flags)
        
        print(f"Running benchmark: {' '.join(cmd)}")
        subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="GLM Distributed Llama Launcher")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models command
    subparsers.add_parser("list", help="List available GLM models")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run GLM model inference")
    inference_parser.add_argument("model", help="Model name")
    inference_parser.add_argument("--prompt", help="Prompt for inference")
    inference_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    inference_parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Run interactive GLM chat")
    chat_parser.add_argument("model", help="Model name")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Chat temperature")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Run GLM worker node")
    worker_parser.add_argument("model", help="Model name")
    worker_parser.add_argument("--nodes", type=int, default=1, help="Total nodes in cluster")
    worker_parser.add_argument("--node-id", type=int, default=0, help="This node's ID")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark GLM model performance")
    benchmark_parser.add_argument("model", help="Model name")
    
    # Setup distributed cluster
    cluster_parser = subparsers.add_parser("setup-cluster", help="Setup distributed inference cluster")
    cluster_parser.add_argument("model", help="Model name")
    cluster_parser.add_argument("--nodes", nargs="+", help="Node IP addresses")
    
    args = parser.parse_args()
    
    launcher = GLMLauncher()
    
    if args.command == "list":
        launcher.list_models()
    elif args.command == "inference":
        launcher.run_inference(args.model, args.prompt, 
                             max_tokens=args.max_tokens, 
                             temperature=args.temperature)
    elif args.command == "chat":
        launcher.run_chat(args.model, temperature=args.temperature)
    elif args.command == "worker":
        launcher.run_worker(args.model, args.nodes)
    elif args.command == "benchmark":
        launcher.benchmark_model(args.model)
    elif args.command == "setup-cluster":
        if not args.nodes:
            print("Error: --nodes required for cluster setup")
            sys.exit(1)
        launcher.setup_distributed_cluster(args.model, args.nodes)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
