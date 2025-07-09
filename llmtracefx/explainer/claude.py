"""
Claude API integration for explaining GPU performance issues
"""
import os
import json
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..profiler.gpu_analyzer import TokenAnalysis, GPUMetrics


@dataclass
class ClaudeExplanation:
    """Claude's explanation of performance issues"""
    token_id: int
    summary: str
    bottleneck_explanation: str
    optimization_suggestions: List[str]
    technical_details: str
    severity: str  # "low", "medium", "high"


class ClaudeExplainer:
    """Use Claude API to explain GPU performance issues"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-opus-20240229"
        
        if not self.api_key:
            raise ValueError("Claude API key not found. Set CLAUDE_API_KEY environment variable.")
    
    async def explain_token_performance(self, analysis: TokenAnalysis) -> ClaudeExplanation:
        """Get Claude's explanation for a single token's performance"""
        prompt = self._build_token_prompt(analysis)
        
        response = await self._call_claude_api(prompt)
        return self._parse_claude_response(analysis.token_id, response)
    
    async def explain_multiple_tokens(self, analyses: List[TokenAnalysis]) -> List[ClaudeExplanation]:
        """Explain performance for multiple tokens concurrently"""
        tasks = [self.explain_token_performance(analysis) for analysis in analyses]
        return await asyncio.gather(*tasks)
    
    def _build_token_prompt(self, analysis: TokenAnalysis) -> str:
        """Build prompt for Claude API"""
        # Format operations for the prompt
        ops_text = ""
        for op in analysis.operations:
            ops_text += f"  - {op.name}: {op.duration:.1f}ms\n"
        
        # Format GPU metrics
        metrics = analysis.gpu_metrics
        
        prompt = f"""You are an expert GPU performance analyst. Analyze this LLM token inference performance and explain the bottlenecks.

TOKEN ANALYSIS:
Token ID: {analysis.token_id}
Token Text: "{analysis.token_text}"
Total Latency: {analysis.total_latency_ms:.1f}ms
Performance Score: {analysis.performance_score:.1f}/100

OPERATIONS:
{ops_text}

GPU METRICS:
- Stall Percentage: {metrics.stall_pct:.1f}%
- Launch Delay: {metrics.launch_delay_ms:.1f}ms
- Memory Latency: {metrics.memory_latency_ms:.1f}ms
- SM Occupancy: {metrics.sm_occupancy_pct:.1f}%
- Cache Hit Rate: {metrics.cache_hit_rate:.1f}%
- Compute Utilization: {metrics.compute_utilization:.1f}%

DETECTED BOTTLENECK: {analysis.bottleneck_type}
OPTIMIZATION FLAGS: {', '.join(analysis.optimization_flags)}

Please provide:
1. A concise summary of the performance issue
2. Technical explanation of why this token was slow
3. Specific optimization recommendations
4. Severity assessment (low/medium/high)

Focus on actionable insights for GPU optimization. Be technical but clear.
"""
        
        return prompt
    
    def _build_sequence_prompt(self, analyses: List[TokenAnalysis]) -> str:
        """Build prompt for analyzing token sequence"""
        total_latency = sum(a.total_latency_ms for a in analyses)
        avg_latency = total_latency / len(analyses)
        
        # Get top bottlenecks
        bottlenecks = {}
        for analysis in analyses:
            bottlenecks[analysis.bottleneck_type] = bottlenecks.get(analysis.bottleneck_type, 0) + 1
        
        top_bottlenecks = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Sample of slowest tokens
        slowest_tokens = sorted(analyses, key=lambda x: x.total_latency_ms, reverse=True)[:5]
        
        slowest_text = ""
        for token in slowest_tokens:
            slowest_text += f"  Token {token.token_id}: {token.total_latency_ms:.1f}ms ({token.bottleneck_type})\n"
        
        prompt = f"""You are an expert GPU performance analyst. Analyze this LLM inference sequence performance.

SEQUENCE OVERVIEW:
Total Tokens: {len(analyses)}
Total Latency: {total_latency:.1f}ms
Average per Token: {avg_latency:.1f}ms

TOP BOTTLENECKS:
{chr(10).join(f'  {name}: {count} tokens' for name, count in top_bottlenecks)}

SLOWEST TOKENS:
{slowest_text}

Please provide:
1. Overall performance assessment
2. Primary bottlenecks affecting the entire sequence
3. Sequence-level optimization strategies
4. Priority recommendations

Focus on system-level optimizations that would improve overall throughput.
"""
        
        return prompt
    
    async def _call_claude_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Claude"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Claude API error {response.status}: {error_text}")
                
                return await response.json()
    
    def _parse_claude_response(self, token_id: int, response: Dict[str, Any]) -> ClaudeExplanation:
        """Parse Claude API response into structured explanation"""
        content = response.get("content", [])
        if not content:
            return self._create_fallback_explanation(token_id)
        
        text = content[0].get("text", "")
        
        # Parse the response text (this is a simplified parser)
        lines = text.split('\n')
        
        summary = ""
        bottleneck_explanation = ""
        optimization_suggestions = []
        technical_details = ""
        severity = "medium"
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if "summary" in line.lower() or "issue" in line.lower():
                current_section = "summary"
                continue
            elif "explanation" in line.lower() or "why" in line.lower():
                current_section = "explanation"
                continue
            elif "optimization" in line.lower() or "recommendation" in line.lower():
                current_section = "optimization"
                continue
            elif "severity" in line.lower():
                current_section = "severity"
                continue
            elif "technical" in line.lower() or "detail" in line.lower():
                current_section = "technical"
                continue
            
            # Extract content based on section
            if current_section == "summary":
                summary += line + " "
            elif current_section == "explanation":
                bottleneck_explanation += line + " "
            elif current_section == "optimization":
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    optimization_suggestions.append(line[1:].strip())
                else:
                    optimization_suggestions.append(line)
            elif current_section == "technical":
                technical_details += line + " "
            elif current_section == "severity":
                if "high" in line.lower():
                    severity = "high"
                elif "low" in line.lower():
                    severity = "low"
                else:
                    severity = "medium"
        
        # Fallback: use the entire response if parsing fails
        if not summary and not bottleneck_explanation:
            summary = text[:200] + "..." if len(text) > 200 else text
            bottleneck_explanation = text
        
        return ClaudeExplanation(
            token_id=token_id,
            summary=summary.strip(),
            bottleneck_explanation=bottleneck_explanation.strip(),
            optimization_suggestions=optimization_suggestions,
            technical_details=technical_details.strip(),
            severity=severity
        )
    
    def _create_fallback_explanation(self, token_id: int) -> ClaudeExplanation:
        """Create fallback explanation if Claude API fails"""
        return ClaudeExplanation(
            token_id=token_id,
            summary="Unable to generate explanation - Claude API error",
            bottleneck_explanation="Analysis failed due to API error",
            optimization_suggestions=["Check API connectivity", "Retry analysis"],
            technical_details="",
            severity="medium"
        )
    
    async def explain_sequence_performance(self, analyses: List[TokenAnalysis]) -> str:
        """Get sequence-level performance explanation"""
        prompt = self._build_sequence_prompt(analyses)
        
        try:
            response = await self._call_claude_api(prompt)
            content = response.get("content", [])
            if content:
                return content[0].get("text", "")
            return "No explanation available"
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def format_explanation_for_display(self, explanation: ClaudeExplanation) -> str:
        """Format explanation for display"""
        output = f"🔍 **Token {explanation.token_id} Analysis**\n\n"
        
        if explanation.summary:
            output += f"**Summary:** {explanation.summary}\n\n"
        
        if explanation.bottleneck_explanation:
            output += f"**Technical Details:** {explanation.bottleneck_explanation}\n\n"
        
        if explanation.optimization_suggestions:
            output += "**Optimization Recommendations:**\n"
            for suggestion in explanation.optimization_suggestions:
                output += f"• {suggestion}\n"
            output += "\n"
        
        output += f"**Severity:** {explanation.severity.upper()}\n"
        
        return output
