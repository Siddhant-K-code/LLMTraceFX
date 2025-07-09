"""
Main CLI entrypoint for LLMTraceFX
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from .profiler.trace_parser import TraceParser
from .profiler.gpu_analyzer import GPUAnalyzer
from .explainer.claude import ClaudeExplainer
from .visualize.flame import FlameGraphGenerator


def create_sample_trace() -> dict:
    """Create sample trace data for testing"""
    return {
        "tokens": [
            {
                "id": 0,
                "text": "Hello",
                "operations": [
                    {"name": "embedding", "start_time": 0, "duration": 2.1},
                    {"name": "rmsnorm", "start_time": 2.1, "duration": 1.8},
                    {"name": "linear", "start_time": 3.9, "duration": 8.2},
                    {"name": "matmul", "start_time": 12.1, "duration": 15.3},
                    {"name": "softmax", "start_time": 27.4, "duration": 3.1}
                ]
            },
            {
                "id": 1,
                "text": "world",
                "operations": [
                    {"name": "embedding", "start_time": 30.5, "duration": 2.3},
                    {"name": "rmsnorm", "start_time": 32.8, "duration": 2.1},
                    {"name": "linear", "start_time": 34.9, "duration": 7.8},
                    {"name": "matmul", "start_time": 42.7, "duration": 18.1},
                    {"name": "kvload", "start_time": 60.8, "duration": 9.2},
                    {"name": "attention", "start_time": 70.0, "duration": 12.4},
                    {"name": "softmax", "start_time": 82.4, "duration": 2.9}
                ]
            },
            {
                "id": 2,
                "text": "!",
                "operations": [
                    {"name": "embedding", "start_time": 85.3, "duration": 1.9},
                    {"name": "rmsnorm", "start_time": 87.2, "duration": 1.7},
                    {"name": "linear", "start_time": 88.9, "duration": 6.8},
                    {"name": "matmul", "start_time": 95.7, "duration": 14.2},
                    {"name": "kvload", "start_time": 109.9, "duration": 11.3},
                    {"name": "attention", "start_time": 121.2, "duration": 16.8},
                    {"name": "softmax", "start_time": 138.0, "duration": 2.1}
                ]
            }
        ]
    }


async def analyze_trace(trace_file: str, gpu_type: str = "A10G", enable_claude: bool = True, 
                       output_dir: str = "output") -> None:
    """Analyze trace file and generate reports"""
    print(f"🔍 Analyzing trace: {trace_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Parse trace
    parser = TraceParser()
    
    try:
        if trace_file == "sample":
            # Use sample data
            trace_data = create_sample_trace()
            tokens = parser.parse_trace_data(trace_data)
            print("📊 Using sample trace data")
        else:
            # Load from file
            tokens = parser.parse_trace_file(trace_file)
            print(f"📊 Loaded {len(tokens)} tokens from {trace_file}")
    except Exception as e:
        print(f"❌ Error parsing trace: {e}")
        return
    
    # Analyze GPU performance
    print(f"🔧 Analyzing GPU performance (GPU: {gpu_type})")
    analyzer = GPUAnalyzer(gpu_type)
    analyses = analyzer.analyze_sequence(tokens)
    
    # Calculate summary stats
    total_latency = sum(a.total_latency_ms for a in analyses)
    avg_latency = total_latency / len(analyses)
    avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
    
    print(f"📈 Analysis complete:")
    print(f"   Total tokens: {len(analyses)}")
    print(f"   Total latency: {total_latency:.1f}ms")
    print(f"   Avg latency per token: {avg_latency:.1f}ms")
    print(f"   Avg performance score: {avg_performance:.1f}/100")
    
    # Generate Claude explanations
    explanations = {}
    if enable_claude:
        try:
            print("🤖 Generating Claude explanations...")
            explainer = ClaudeExplainer()
            
            for analysis in analyses:
                try:
                    explanation = await explainer.explain_token_performance(analysis)
                    explanations[analysis.token_id] = explainer.format_explanation_for_display(explanation)
                    print(f"   Token {analysis.token_id}: ✓")
                except Exception as e:
                    explanations[analysis.token_id] = f"Error: {str(e)}"
                    print(f"   Token {analysis.token_id}: ❌ {str(e)}")
        except Exception as e:
            print(f"⚠️  Claude explanations disabled: {e}")
            enable_claude = False
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    visualizer = FlameGraphGenerator()
    
    try:
        # Generate dashboard
        dashboard_html = visualizer.generate_comprehensive_dashboard(analyses)
        dashboard_path = output_path / "dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)
        print(f"   Dashboard: {dashboard_path}")
        
        # Generate individual charts
        flame_graph = visualizer.generate_token_flame_graph(analyses)
        flame_path = output_path / "flame_graph.html"
        with open(flame_path, "w") as f:
            f.write(flame_graph)
        print(f"   Flame graph: {flame_path}")
        
        # Export data
        export_json = visualizer.export_data_json(analyses)
        export_path = output_path / "analysis_data.json"
        with open(export_path, "w") as f:
            f.write(export_json)
        print(f"   Data export: {export_path}")
        
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")
    
    # Generate text report
    print("📝 Generating text report...")
    report_path = output_path / "report.txt"
    with open(report_path, "w") as f:
        f.write("LLMTraceFX Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Trace File: {trace_file}\n")
        f.write(f"GPU Type: {gpu_type}\n")
        f.write(f"Analysis Time: {total_latency:.1f}ms\n")
        f.write(f"Average Performance Score: {avg_performance:.1f}/100\n\n")
        
        # Bottleneck summary
        bottleneck_counts = {}
        for analysis in analyses:
            bottleneck_counts[analysis.bottleneck_type] = bottleneck_counts.get(analysis.bottleneck_type, 0) + 1
        
        f.write("Bottleneck Distribution:\n")
        for bottleneck, count in sorted(bottleneck_counts.items()):
            f.write(f"  {bottleneck}: {count} tokens\n")
        f.write("\n")
        
        # Token details
        f.write("Token Analysis:\n")
        f.write("-" * 30 + "\n")
        
        for analysis in analyses:
            f.write(f"\nToken {analysis.token_id}: \"{analysis.token_text}\"\n")
            f.write(f"  Latency: {analysis.total_latency_ms:.1f}ms\n")
            f.write(f"  Performance Score: {analysis.performance_score:.1f}/100\n")
            f.write(f"  Bottleneck: {analysis.bottleneck_type}\n")
            f.write(f"  Optimization Flags: {', '.join(analysis.optimization_flags)}\n")
            
            f.write("  Operations:\n")
            for op in analysis.operations:
                f.write(f"    {op.name}: {op.duration:.1f}ms\n")
            
            f.write("  GPU Metrics:\n")
            metrics = analysis.gpu_metrics
            f.write(f"    Stall %: {metrics.stall_pct:.1f}%\n")
            f.write(f"    Launch Delay: {metrics.launch_delay_ms:.1f}ms\n")
            f.write(f"    SM Occupancy: {metrics.sm_occupancy_pct:.1f}%\n")
            f.write(f"    Cache Hit Rate: {metrics.cache_hit_rate:.1f}%\n")
            
            # Include Claude explanation if available
            if enable_claude and analysis.token_id in explanations:
                f.write("\n  Claude Explanation:\n")
                explanation_lines = explanations[analysis.token_id].split('\n')
                for line in explanation_lines:
                    f.write(f"    {line}\n")
            
            f.write("\n")
    
    print(f"   Text report: {report_path}")
    print(f"\n✅ Analysis complete! Check the {output_dir} directory for results.")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="LLMTraceFX - GPU-level LLM inference profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze sample trace
  python -m llmtracefx.main --trace sample
  
  # Analyze trace file
  python -m llmtracefx.main --trace test_traces/sample_vllm_trace.json
  
  # Analyze with specific GPU type
  python -m llmtracefx.main --trace sample --gpu-type H100
  
  # Analyze without Claude explanations
  python -m llmtracefx.main --trace sample --no-claude
        """
    )
    
    parser.add_argument(
        "--trace", 
        type=str, 
        default="sample",
        help="Trace file to analyze (use 'sample' for built-in sample data)"
    )
    
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="A10G",
        choices=["A10G", "H100", "A100"],
        help="GPU type for analysis"
    )
    
    parser.add_argument(
        "--no-claude",
        action="store_true",
        help="Disable Claude AI explanations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample trace file and exit"
    )
    
    args = parser.parse_args()
    
    # Create sample trace if requested
    if args.create_sample:
        sample_data = create_sample_trace()
        sample_path = Path("test_traces/sample_vllm_trace.json")
        sample_path.parent.mkdir(exist_ok=True)
        
        with open(sample_path, "w") as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Sample trace created: {sample_path}")
        return
    
    # Run analysis
    enable_claude = not args.no_claude
    
    try:
        asyncio.run(analyze_trace(
            args.trace,
            args.gpu_type,
            enable_claude,
            args.output_dir
        ))
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
