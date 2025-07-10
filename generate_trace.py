#!/usr/bin/env python3
"""
Generate custom trace files for LLMTraceFX
"""

import json
import random
import argparse
from datetime import datetime

def generate_realistic_timing(op_name, performance_profile="balanced"):
    """Generate realistic timing for GPU operations"""
    
    # Base timings in milliseconds for different operation types
    base_timings = {
        "embedding": {"min": 1.0, "max": 4.0},
        "rmsnorm": {"min": 0.8, "max": 3.0},
        "layernorm": {"min": 0.8, "max": 3.0},
        "linear": {"min": 5.0, "max": 15.0},
        "matmul": {"min": 8.0, "max": 35.0},
        "attention": {"min": 10.0, "max": 30.0},
        "softmax": {"min": 1.5, "max": 6.0},
        "kvload": {"min": 3.0, "max": 20.0},
        "kvstore": {"min": 2.0, "max": 15.0}
    }
    
    # Performance multipliers based on profile
    multipliers = {
        "optimized": 0.6,   # Fast, well-optimized
        "balanced": 1.0,    # Normal performance
        "memory_bound": 1.8, # Slow due to memory issues
        "compute_bound": 1.5 # Slow due to compute issues
    }
    
    base = base_timings.get(op_name, {"min": 2.0, "max": 10.0})
    multiplier = multipliers.get(performance_profile, 1.0)
    
    # Add some randomness
    duration = random.uniform(base["min"], base["max"]) * multiplier
    return round(duration, 1)

def generate_trace(tokens, performance_profile="balanced", operations_per_token=None):
    """Generate a complete trace file"""
    
    if operations_per_token is None:
        operations_per_token = ["embedding", "rmsnorm", "linear", "matmul", "attention", "softmax"]
    
    trace = {"tokens": []}
    
    for i, token_text in enumerate(tokens):
        current_time = 0.0
        operations = []
        
        # Determine operations for this token
        if i == 0:
            # First token usually doesn't have KV operations
            ops = [op for op in operations_per_token if not op.startswith("kv")]
        else:
            # Subsequent tokens may have KV operations
            ops = operations_per_token.copy()
            if random.random() > 0.7:  # 70% chance to add KV operations
                if "kvload" not in ops:
                    ops.insert(-2, "kvload")  # Add before attention
        
        for op_name in ops:
            duration = generate_realistic_timing(op_name, performance_profile)
            operations.append({
                "name": op_name,
                "start_time": round(current_time, 1),
                "duration": duration
            })
            current_time += duration
        
        trace["tokens"].append({
            "id": i,
            "text": token_text,
            "operations": operations
        })
    
    return trace

def main():
    parser = argparse.ArgumentParser(description='Generate LLMTraceFX trace files')
    parser.add_argument('--tokens', nargs='+', default=["Hello", "world", "from", "LLM", "!"],
                        help='List of tokens to include in trace')
    parser.add_argument('--profile', choices=['optimized', 'balanced', 'memory_bound', 'compute_bound'],
                        default='balanced', help='Performance profile')
    parser.add_argument('--output', default=None, help='Output filename')
    parser.add_argument('--operations', nargs='+', 
                        default=["embedding", "rmsnorm", "linear", "matmul", "attention", "softmax"],
                        help='List of operations per token')
    
    args = parser.parse_args()
    
    # Generate trace
    trace = generate_trace(args.tokens, args.profile, args.operations)
    
    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"trace_{args.profile}_{timestamp}.json"
    
    # Save trace
    with open(args.output, 'w') as f:
        json.dump(trace, f, indent=2)
    
    print(f"✅ Generated trace file: {args.output}")
    print(f"   Profile: {args.profile}")
    print(f"   Tokens: {len(args.tokens)}")
    print(f"   Operations: {args.operations}")
    
    # Calculate total latency
    total_latency = 0
    for token in trace["tokens"]:
        if token["operations"]:
            last_op = token["operations"][-1]
            token_end_time = last_op["start_time"] + last_op["duration"]
            total_latency = max(total_latency, token_end_time)
    
    print(f"   Total latency: {total_latency:.1f}ms")
    print(f"   Avg latency per token: {total_latency/len(args.tokens):.1f}ms")

if __name__ == "__main__":
    main()
