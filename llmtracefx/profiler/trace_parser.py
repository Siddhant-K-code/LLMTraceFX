"""
Trace parser for LLM inference logs (vLLM format)
"""
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Operation:
    """Represents a GPU operation/kernel"""
    name: str
    start_time: float
    duration: float
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TokenTrace:
    """Per-token trace with operations"""
    token_id: int
    token_text: str
    total_latency: float
    operations: List[Operation]
    start_time: float
    end_time: float


class TraceParser:
    """Parse vLLM trace logs into token-level operations"""
    
    def __init__(self):
        self.known_ops = {
            "rmsnorm", "layernorm", "linear", "matmul", "softmax", 
            "kvload", "kvstore", "attention", "activation", "embedding"
        }
    
    def parse_trace_file(self, filepath: str) -> List[TokenTrace]:
        """Parse trace file and return token-level breakdown"""
        with open(filepath, 'r') as f:
            trace_data = json.load(f)
        
        return self.parse_trace_data(trace_data)
    
    def parse_trace_data(self, trace_data: Dict[str, Any]) -> List[TokenTrace]:
        """Parse trace data structure"""
        if "tokens" in trace_data:
            return self._parse_vllm_format(trace_data)
        elif "events" in trace_data:
            return self._parse_event_format(trace_data)
        else:
            return self._parse_generic_format(trace_data)
    
    def _parse_vllm_format(self, data: Dict[str, Any]) -> List[TokenTrace]:
        """Parse vLLM-specific trace format"""
        tokens = []
        
        for token_data in data.get("tokens", []):
            token_id = token_data.get("id", 0)
            token_text = token_data.get("text", "")
            
            operations = []
            total_latency = 0
            start_time = float('inf')
            end_time = 0
            
            for op_data in token_data.get("operations", []):
                op = Operation(
                    name=self._normalize_op_name(op_data.get("name", "")),
                    start_time=op_data.get("start_time", 0),
                    duration=op_data.get("duration", 0),
                    dependencies=op_data.get("dependencies", []),
                    metadata=op_data.get("metadata", {})
                )
                operations.append(op)
                
                total_latency += op.duration
                start_time = min(start_time, op.start_time)
                end_time = max(end_time, op.start_time + op.duration)
            
            token_trace = TokenTrace(
                token_id=token_id,
                token_text=token_text,
                total_latency=total_latency,
                operations=operations,
                start_time=start_time if start_time != float('inf') else 0,
                end_time=end_time
            )
            tokens.append(token_trace)
        
        return tokens
    
    def _parse_event_format(self, data: Dict[str, Any]) -> List[TokenTrace]:
        """Parse event-based trace format"""
        events = data.get("events", [])
        token_map = {}
        
        for event in events:
            token_id = event.get("token_id", 0)
            if token_id not in token_map:
                token_map[token_id] = {
                    "operations": [],
                    "token_text": event.get("token_text", "")
                }
            
            op = Operation(
                name=self._normalize_op_name(event.get("op_name", "")),
                start_time=event.get("timestamp", 0),
                duration=event.get("duration", 0),
                metadata=event.get("metadata", {})
            )
            token_map[token_id]["operations"].append(op)
        
        tokens = []
        for token_id, token_data in token_map.items():
            operations = token_data["operations"]
            total_latency = sum(op.duration for op in operations)
            start_time = min(op.start_time for op in operations) if operations else 0
            end_time = max(op.start_time + op.duration for op in operations) if operations else 0
            
            token_trace = TokenTrace(
                token_id=token_id,
                token_text=token_data["token_text"],
                total_latency=total_latency,
                operations=operations,
                start_time=start_time,
                end_time=end_time
            )
            tokens.append(token_trace)
        
        return sorted(tokens, key=lambda x: x.token_id)
    
    def _parse_generic_format(self, data: Dict[str, Any]) -> List[TokenTrace]:
        """Parse generic trace format - create synthetic data if needed"""
        # For demo purposes, create sample token traces
        num_tokens = data.get("num_tokens", 10)
        tokens = []
        
        for i in range(num_tokens):
            operations = [
                Operation("rmsnorm", i * 50, 2.3),
                Operation("matmul", i * 50 + 5, 12.4),
                Operation("kvload", i * 50 + 20, 9.1),
                Operation("softmax", i * 50 + 35, 3.2)
            ]
            
            token_trace = TokenTrace(
                token_id=i,
                token_text=f"token_{i}",
                total_latency=sum(op.duration for op in operations),
                operations=operations,
                start_time=i * 50,
                end_time=i * 50 + 45
            )
            tokens.append(token_trace)
        
        return tokens
    
    def _normalize_op_name(self, name: str) -> str:
        """Normalize operation names to standard set"""
        name_lower = name.lower().replace("_", "").replace("-", "")
        
        for known_op in self.known_ops:
            if known_op in name_lower:
                return known_op
        
        return name_lower
    
    def get_summary_stats(self, tokens: List[TokenTrace]) -> Dict[str, Any]:
        """Get summary statistics for all tokens"""
        if not tokens:
            return {}
        
        total_latency = sum(token.total_latency for token in tokens)
        avg_latency = total_latency / len(tokens)
        
        op_counts = {}
        op_times = {}
        
        for token in tokens:
            for op in token.operations:
                op_counts[op.name] = op_counts.get(op.name, 0) + 1
                op_times[op.name] = op_times.get(op.name, 0) + op.duration
        
        return {
            "total_tokens": len(tokens),
            "total_latency_ms": total_latency,
            "avg_latency_per_token_ms": avg_latency,
            "operation_counts": op_counts,
            "operation_total_times": op_times,
            "timeline_start": min(token.start_time for token in tokens),
            "timeline_end": max(token.end_time for token in tokens)
        }
