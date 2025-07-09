"""
Configuration utilities for LLMTraceFX
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for LLMTraceFX"""
    
    # Claude API settings
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-opus-20240229"
    claude_max_tokens: int = 1000
    
    # GPU settings
    default_gpu_type: str = "A10G"
    
    # Analysis settings
    enable_claude_by_default: bool = True
    max_tokens_per_analysis: int = 1000
    
    # Visualization settings
    dashboard_port: int = 8000
    enable_plotly_cdn: bool = True
    
    # Output settings
    default_output_dir: str = "output"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            claude_api_key=os.environ.get("CLAUDE_API_KEY"),
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229"),
            claude_max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "1000")),
            default_gpu_type=os.environ.get("DEFAULT_GPU_TYPE", "A10G"),
            enable_claude_by_default=os.environ.get("ENABLE_CLAUDE", "true").lower() == "true",
            max_tokens_per_analysis=int(os.environ.get("MAX_TOKENS_PER_ANALYSIS", "1000")),
            dashboard_port=int(os.environ.get("DASHBOARD_PORT", "8000")),
            enable_plotly_cdn=os.environ.get("ENABLE_PLOTLY_CDN", "true").lower() == "true",
            default_output_dir=os.environ.get("DEFAULT_OUTPUT_DIR", "output")
        )


# Global config instance
config = Config.from_env()
