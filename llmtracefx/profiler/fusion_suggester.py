"""
Kernel fusion suggestions for GPU optimization
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .trace_parser import Operation, TokenTrace
from .gpu_analyzer import TokenAnalysis


@dataclass
class FusionOpportunity:
    """Represents a kernel fusion opportunity"""
    operation_names: List[str]
    operation_indices: List[int]
    fusion_type: str
    estimated_speedup: float
    confidence: float
    description: str
    requirements: List[str]
    potential_issues: List[str]


class FusionSuggester:
    """Analyze and suggest kernel fusion opportunities"""
    
    def __init__(self):
        # Define fusion patterns and their characteristics
        self.fusion_patterns = {
            "elementwise_chain": {
                "operations": ["rmsnorm", "activation", "layernorm"],
                "description": "Fuse consecutive elementwise operations",
                "speedup_factor": 1.8,
                "confidence": 0.9
            },
            "linear_norm": {
                "operations": ["linear", "rmsnorm"],
                "description": "Fuse linear transformation with normalization",
                "speedup_factor": 1.5,
                "confidence": 0.8
            },
            "norm_linear": {
                "operations": ["rmsnorm", "linear"],
                "description": "Fuse normalization with linear transformation",
                "speedup_factor": 1.6,
                "confidence": 0.85
            },
            "attention_qkv": {
                "operations": ["linear", "linear", "linear"],
                "description": "Fuse Query, Key, Value linear projections",
                "speedup_factor": 2.2,
                "confidence": 0.9
            },
            "matmul_softmax": {
                "operations": ["matmul", "softmax"],
                "description": "Fuse matrix multiplication with softmax",
                "speedup_factor": 1.4,
                "confidence": 0.75
            },
            "embedding_norm": {
                "operations": ["embedding", "rmsnorm"],
                "description": "Fuse embedding lookup with normalization",
                "speedup_factor": 1.3,
                "confidence": 0.7
            }
        }
        
        # Define operation compatibility
        self.compatible_ops = {
            "rmsnorm": ["linear", "embedding", "activation"],
            "layernorm": ["linear", "embedding", "activation"],
            "linear": ["rmsnorm", "layernorm", "activation", "softmax"],
            "matmul": ["softmax", "activation"],
            "softmax": ["linear"],
            "embedding": ["rmsnorm", "layernorm"],
            "activation": ["linear", "rmsnorm", "layernorm"]
        }
    
    def analyze_token_fusion(self, analysis: TokenAnalysis) -> List[FusionOpportunity]:
        """Analyze fusion opportunities for a single token"""
        opportunities = []
        operations = analysis.operations
        
        # Check for pattern-based fusion opportunities
        pattern_opportunities = self._find_pattern_opportunities(operations)
        opportunities.extend(pattern_opportunities)
        
        # Check for adjacent operation fusion
        adjacent_opportunities = self._find_adjacent_opportunities(operations)
        opportunities.extend(adjacent_opportunities)
        
        # Check for small kernel fusion
        small_kernel_opportunities = self._find_small_kernel_opportunities(operations)
        opportunities.extend(small_kernel_opportunities)
        
        # Filter and rank opportunities
        filtered_opportunities = self._filter_and_rank(opportunities, analysis)
        
        return filtered_opportunities
    
    def _find_pattern_opportunities(self, operations: List[Operation]) -> List[FusionOpportunity]:
        """Find fusion opportunities based on known patterns"""
        opportunities = []
        
        for pattern_name, pattern_info in self.fusion_patterns.items():
            pattern_ops = pattern_info["operations"]
            
            # Look for exact pattern matches
            for i in range(len(operations) - len(pattern_ops) + 1):
                op_sequence = [operations[i + j].name for j in range(len(pattern_ops))]
                
                if op_sequence == pattern_ops:
                    # Calculate estimated speedup based on current timings
                    total_time = sum(operations[i + j].duration for j in range(len(pattern_ops)))
                    estimated_speedup = total_time * (pattern_info["speedup_factor"] - 1)
                    
                    opportunity = FusionOpportunity(
                        operation_names=op_sequence,
                        operation_indices=list(range(i, i + len(pattern_ops))),
                        fusion_type=pattern_name,
                        estimated_speedup=estimated_speedup,
                        confidence=pattern_info["confidence"],
                        description=pattern_info["description"],
                        requirements=self._get_fusion_requirements(pattern_name),
                        potential_issues=self._get_potential_issues(pattern_name)
                    )
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _find_adjacent_opportunities(self, operations: List[Operation]) -> List[FusionOpportunity]:
        """Find fusion opportunities between adjacent operations"""
        opportunities = []
        
        for i in range(len(operations) - 1):
            op1 = operations[i]
            op2 = operations[i + 1]
            
            # Check if operations are compatible for fusion
            if self._are_operations_compatible(op1.name, op2.name):
                # Estimate fusion benefit
                launch_overhead = min(op1.duration, op2.duration) * 0.1  # Assume 10% launch overhead
                memory_benefit = (op1.duration + op2.duration) * 0.05  # Assume 5% memory optimization
                
                estimated_speedup = launch_overhead + memory_benefit
                
                opportunity = FusionOpportunity(
                    operation_names=[op1.name, op2.name],
                    operation_indices=[i, i + 1],
                    fusion_type="adjacent_fusion",
                    estimated_speedup=estimated_speedup,
                    confidence=0.6,
                    description=f"Fuse adjacent {op1.name} and {op2.name} operations",
                    requirements=["Compatible memory layouts", "Similar thread requirements"],
                    potential_issues=["Increased register pressure", "Reduced parallelism"]
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _find_small_kernel_opportunities(self, operations: List[Operation]) -> List[FusionOpportunity]:
        """Find opportunities to fuse small kernels"""
        opportunities = []
        small_ops = []
        
        # Find small operations (< 5ms)
        for i, op in enumerate(operations):
            if op.duration < 5.0:
                small_ops.append((i, op))
        
        # Group consecutive small operations
        if len(small_ops) >= 2:
            consecutive_groups = []
            current_group = [small_ops[0]]
            
            for i in range(1, len(small_ops)):
                if small_ops[i][0] == small_ops[i-1][0] + 1:  # Consecutive indices
                    current_group.append(small_ops[i])
                else:
                    if len(current_group) >= 2:
                        consecutive_groups.append(current_group)
                    current_group = [small_ops[i]]
            
            if len(current_group) >= 2:
                consecutive_groups.append(current_group)
            
            # Create fusion opportunities for consecutive small operations
            for group in consecutive_groups:
                indices = [op_info[0] for op_info in group]
                ops = [op_info[1] for op_info in group]
                
                # Estimate speedup from reduced launch overhead
                total_time = sum(op.duration for op in ops)
                estimated_speedup = total_time * 0.2  # Assume 20% improvement for small kernels
                
                opportunity = FusionOpportunity(
                    operation_names=[op.name for op in ops],
                    operation_indices=indices,
                    fusion_type="small_kernel_fusion",
                    estimated_speedup=estimated_speedup,
                    confidence=0.7,
                    description=f"Fuse {len(ops)} small consecutive operations",
                    requirements=["Compatible data types", "Similar memory access patterns"],
                    potential_issues=["Increased kernel complexity", "Potential register spilling"]
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def _are_operations_compatible(self, op1_name: str, op2_name: str) -> bool:
        """Check if two operations are compatible for fusion"""
        return op2_name in self.compatible_ops.get(op1_name, [])
    
    def _get_fusion_requirements(self, fusion_type: str) -> List[str]:
        """Get requirements for a specific fusion type"""
        requirements = {
            "elementwise_chain": [
                "Similar tensor shapes",
                "Compatible data types",
                "Sufficient GPU memory"
            ],
            "linear_norm": [
                "Compatible matrix dimensions",
                "Sufficient shared memory",
                "Tensor core availability"
            ],
            "norm_linear": [
                "Compatible tensor layouts",
                "Sufficient registers",
                "Memory coalescing support"
            ],
            "attention_qkv": [
                "Identical input tensor",
                "Compatible output dimensions",
                "Sufficient GPU memory bandwidth"
            ],
            "matmul_softmax": [
                "Compatible tensor shapes",
                "Sufficient shared memory",
                "Numerical stability considerations"
            ],
            "embedding_norm": [
                "Compatible embedding dimensions",
                "Sufficient GPU memory",
                "Efficient indexing support"
            ]
        }
        
        return requirements.get(fusion_type, ["General compatibility checks"])
    
    def _get_potential_issues(self, fusion_type: str) -> List[str]:
        """Get potential issues for a specific fusion type"""
        issues = {
            "elementwise_chain": [
                "Increased register pressure",
                "Reduced flexibility for optimization"
            ],
            "linear_norm": [
                "Complexity in error handling",
                "Potential numerical instability"
            ],
            "norm_linear": [
                "Memory access pattern changes",
                "Debugging complexity"
            ],
            "attention_qkv": [
                "Increased kernel size",
                "Potential cache pressure"
            ],
            "matmul_softmax": [
                "Numerical precision concerns",
                "Limited reusability"
            ],
            "embedding_norm": [
                "Irregular memory access",
                "Potential bank conflicts"
            ]
        }
        
        return issues.get(fusion_type, ["General fusion complexity"])
    
    def _filter_and_rank(self, opportunities: List[FusionOpportunity], 
                        analysis: TokenAnalysis) -> List[FusionOpportunity]:
        """Filter and rank fusion opportunities"""
        # Filter out low-confidence opportunities
        filtered = [opp for opp in opportunities if opp.confidence > 0.5]
        
        # Calculate priority score
        for opp in filtered:
            # Base score from estimated speedup
            speedup_score = opp.estimated_speedup
            
            # Confidence multiplier
            confidence_multiplier = opp.confidence
            
            # Complexity penalty (more operations = more complex)
            complexity_penalty = len(opp.operation_names) * 0.1
            
            # Performance score bonus (lower performance = higher priority)
            performance_bonus = (100 - analysis.performance_score) * 0.01
            
            opp.priority_score = (speedup_score * confidence_multiplier + 
                                performance_bonus - complexity_penalty)
        
        # Sort by priority score
        filtered.sort(key=lambda x: x.priority_score, reverse=True)
        
        return filtered
    
    def analyze_sequence_fusion(self, analyses: List[TokenAnalysis]) -> Dict[str, Any]:
        """Analyze fusion opportunities across a sequence of tokens"""
        all_opportunities = []
        
        for analysis in analyses:
            token_opportunities = self.analyze_token_fusion(analysis)
            all_opportunities.extend(token_opportunities)
        
        # Aggregate statistics
        fusion_stats = {
            "total_opportunities": len(all_opportunities),
            "fusion_types": {},
            "total_estimated_speedup": 0,
            "avg_confidence": 0,
            "top_opportunities": []
        }
        
        if all_opportunities:
            # Count fusion types
            for opp in all_opportunities:
                fusion_stats["fusion_types"][opp.fusion_type] = (
                    fusion_stats["fusion_types"].get(opp.fusion_type, 0) + 1
                )
            
            # Calculate totals
            fusion_stats["total_estimated_speedup"] = sum(
                opp.estimated_speedup for opp in all_opportunities
            )
            fusion_stats["avg_confidence"] = sum(
                opp.confidence for opp in all_opportunities
            ) / len(all_opportunities)
            
            # Get top opportunities
            sorted_opportunities = sorted(
                all_opportunities, 
                key=lambda x: x.estimated_speedup, 
                reverse=True
            )
            fusion_stats["top_opportunities"] = sorted_opportunities[:5]
        
        return fusion_stats
    
    def generate_fusion_report(self, analyses: List[TokenAnalysis]) -> str:
        """Generate a human-readable fusion report"""
        sequence_stats = self.analyze_sequence_fusion(analyses)
        
        report = "🔗 Kernel Fusion Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total Fusion Opportunities: {sequence_stats['total_opportunities']}\n"
        report += f"Estimated Total Speedup: {sequence_stats['total_estimated_speedup']:.1f}ms\n"
        report += f"Average Confidence: {sequence_stats['avg_confidence']:.1f}%\n\n"
        
        if sequence_stats['fusion_types']:
            report += "Fusion Type Distribution:\n"
            for fusion_type, count in sequence_stats['fusion_types'].items():
                report += f"  {fusion_type}: {count} opportunities\n"
            report += "\n"
        
        if sequence_stats['top_opportunities']:
            report += "Top Fusion Opportunities:\n"
            report += "-" * 30 + "\n"
            
            for i, opp in enumerate(sequence_stats['top_opportunities'], 1):
                report += f"\n{i}. {opp.description}\n"
                report += f"   Operations: {' → '.join(opp.operation_names)}\n"
                report += f"   Estimated Speedup: {opp.estimated_speedup:.1f}ms\n"
                report += f"   Confidence: {opp.confidence:.1f}%\n"
                
                if opp.requirements:
                    report += f"   Requirements: {', '.join(opp.requirements)}\n"
                
                if opp.potential_issues:
                    report += f"   Potential Issues: {', '.join(opp.potential_issues)}\n"
        
        return report
