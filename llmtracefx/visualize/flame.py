"""
Flame graph visualization for token-level GPU performance
"""

import json
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..brand import (
    CHART_SEQUENCE,
    HEATMAP_SCALE,
    LOCKUP_SVG,
    PLOT_ANNOTATION,
    PLOT_LAYOUT,
    TOKENS_CSS,
)
from ..profiler.gpu_analyzer import TokenAnalysis


class FlameGraphGenerator:
    """Generate flame graphs and performance visualizations"""

    def __init__(self) -> None:
        # Operations are keyed to the brand series so the same operation is
        # the same colour in every chart, and so nothing drifts off-palette.
        self.color_map = dict(
            zip(
                (
                    "rmsnorm",
                    "layernorm",
                    "linear",
                    "matmul",
                    "softmax",
                    "kvload",
                    "kvstore",
                    "attention",
                    "activation",
                    "embedding",
                ),
                CHART_SEQUENCE,
                strict=True,
            )
        )

    def _themed(self, fig: Any, div_id: str) -> str:
        """Put a figure on the sheet and render it.

        Applied at the one point every chart exits, so a new chart cannot be
        added later and quietly arrive wearing the library's default theme.
        """
        fig.update_layout(**PLOT_LAYOUT)
        # Subplot titles are paper annotations, not part of layout, so they
        # survive the theme above and would otherwise arrive as a second
        # heading competing with the panel heading in the page shell.
        fig.update_annotations(**PLOT_ANNOTATION)
        # Without this the figure keeps the library's fixed default width and
        # the browser scales the whole plot down to fit a narrow column, which
        # shrinks the tick labels and in-bar readings past legibility. Filling
        # the container instead keeps type at its intended size.
        html: str = fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            div_id=div_id,
            config={"responsive": True, "showSendToCloud": False},
        )
        return html

    def generate_token_flame_graph(self, analyses: list[TokenAnalysis]) -> str:
        """Generate flame graph showing token vs operations timeline"""
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])

        for analysis in analyses:
            for op in analysis.operations:
                color = self.color_map.get(op.name, "#4a5157")

                # Add operation bar
                fig.add_trace(
                    go.Bar(
                        x=[op.duration],
                        y=[f"Token {analysis.token_id}"],
                        name=op.name,
                        orientation="h",
                        marker_color=color,
                        text=f"{op.name}: {op.duration:.1f}ms",
                        textposition="inside",
                        hovertemplate=f"<b>{op.name}</b><br>"
                        + f"Duration: {op.duration:.1f}ms<br>"
                        + f"Token: {analysis.token_id}<br>"
                        + f"Performance Score: {analysis.performance_score:.1f}<extra></extra>",
                        showlegend=True if analysis.token_id == 0 else False,
                    )
                )

        # Update layout
        fig.update_layout(
            xaxis_title="Time (ms)",
            yaxis_title="Tokens",
            barmode="stack",
            height=max(400, len(analyses) * 40),
            showlegend=True,
        )

        return self._themed(fig, "flame-graph")

    def generate_bottleneck_distribution(self, analyses: list[TokenAnalysis]) -> str:
        """Generate bottleneck distribution chart"""
        bottleneck_counts: dict[str, int] = {}
        for analysis in analyses:
            bottleneck_counts[analysis.bottleneck_type] = (
                bottleneck_counts.get(analysis.bottleneck_type, 0) + 1
            )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(bottleneck_counts.keys()),
                    y=list(bottleneck_counts.values()),
                    marker_color=list(CHART_SEQUENCE[:6]),
                )
            ]
        )

        fig.update_layout(
            xaxis_title="Bottleneck Type", yaxis_title="Number of Tokens", height=400
        )

        return self._themed(fig, "bottleneck-chart")

    def generate_performance_heatmap(self, analyses: list[TokenAnalysis]) -> str:
        """Generate performance heatmap"""
        # Create matrix data
        operations = sorted(
            {op.name for analysis in analyses for op in analysis.operations}
        )

        # Create matrix
        matrix: list[list[float]] = []
        token_ids: list[str] = []

        for analysis in analyses:
            row: list[float] = []
            token_ids.append(f"Token {analysis.token_id}")

            for op_name in operations:
                # Find operation duration
                duration = 0.0
                for op in analysis.operations:
                    if op.name == op_name:
                        duration = op.duration
                        break
                row.append(duration)
            matrix.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=operations,
                y=token_ids,
                colorscale=[list(stop) for stop in HEATMAP_SCALE],
                hovertemplate="<b>%{x}</b><br>%{y}<br>Duration: %{z:.1f}ms<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="GPU Operations",
            yaxis_title="Tokens",
            height=max(400, len(analyses) * 20),
        )

        return self._themed(fig, "heatmap")

    def generate_latency_trend(self, analyses: list[TokenAnalysis]) -> str:
        """Generate latency trend chart"""
        token_ids = [analysis.token_id for analysis in analyses]
        latencies = [analysis.total_latency_ms for analysis in analyses]
        performance_scores = [analysis.performance_score for analysis in analyses]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["TOKEN LATENCY TREND", "PERFORMANCE SCORE TREND"],
            vertical_spacing=0.1,
        )

        # Latency trend
        fig.add_trace(
            go.Scatter(
                x=token_ids,
                y=latencies,
                mode="lines+markers",
                name="Latency (ms)",
                line={"color": CHART_SEQUENCE[0], "width": 2},
                marker={"size": 6},
            ),
            row=1,
            col=1,
        )

        # Performance score trend
        fig.add_trace(
            go.Scatter(
                x=token_ids,
                y=performance_scores,
                mode="lines+markers",
                name="Performance Score",
                line={"color": CHART_SEQUENCE[1], "width": 2},
                marker={"size": 6},
            ),
            row=2,
            col=1,
        )

        fig.update_layout(height=600, showlegend=True)

        fig.update_xaxes(title_text="Token ID", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=2, col=1)

        return self._themed(fig, "trend-chart")

    def generate_gpu_metrics_radar(self, analysis: TokenAnalysis) -> str:
        """Generate radar chart for GPU metrics"""
        metrics = analysis.gpu_metrics

        categories = [
            f"{metrics.occupancy_label} %",
            "Cache Hit Rate %",
            "Compute Utilization %",
            "Memory Efficiency %",
            "Launch Efficiency %",
        ]

        values = [
            metrics.sm_occupancy_pct,
            metrics.cache_hit_rate,
            metrics.compute_utilization,
            100 - (metrics.stall_pct),  # Memory efficiency
            100 - min(metrics.launch_delay_ms * 10, 100),  # Launch efficiency
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=f"Token {analysis.token_id}",
                line_color=CHART_SEQUENCE[1],
            )
        )

        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            showlegend=True,
            title=f"GPU Metrics Profile - Token {analysis.token_id}",
            height=500,
        )

        return self._themed(fig, "radar-chart")

    def generate_operation_breakdown(self, analysis: TokenAnalysis) -> str:
        """Generate pie chart for operation breakdown"""
        op_names = [op.name for op in analysis.operations]
        op_durations = [op.duration for op in analysis.operations]

        colors = [self.color_map.get(name, "#95A5A6") for name in op_names]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=op_names,
                    values=op_durations,
                    hole=0.3,
                    marker_colors=colors,
                    textinfo="label+percent+value",
                    texttemplate="%{label}<br>%{percent}<br>%{value:.1f}ms",
                )
            ]
        )

        fig.update_layout(
            title=f"Operation Breakdown - Token {analysis.token_id}", height=400
        )

        return self._themed(fig, "breakdown-chart")

    def generate_comprehensive_dashboard(self, analyses: list[TokenAnalysis]) -> str:
        """Generate comprehensive HTML dashboard"""
        if not analyses:
            return "<html><body><h1>No data to display</h1></body></html>"

        # Generate all charts
        flame_graph = self.generate_token_flame_graph(analyses)
        bottleneck_dist = self.generate_bottleneck_distribution(analyses)
        heatmap = self.generate_performance_heatmap(analyses)
        trend_chart = self.generate_latency_trend(analyses)

        # Sample detailed analysis for first token
        sample_radar = self.generate_gpu_metrics_radar(analyses[0])
        sample_breakdown = self.generate_operation_breakdown(analyses[0])

        # Generate summary stats
        total_latency = sum(a.total_latency_ms for a in analyses)
        avg_latency = total_latency / len(analyses)
        avg_performance = sum(a.performance_score for a in analyses) / len(analyses)

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="color-scheme" content="light">
            <title>GPU performance dashboard - LLMTraceFX</title>
            <style>
                {TOKENS_CSS}
                * {{ box-sizing: border-box; }}
                body {{
                    font-family: var(--sans);
                    margin: 0;
                    padding: clamp(16px, 3vw, 40px) clamp(12px, 3vw, 36px) 64px;
                    background-color: var(--field);
                    background-image:
                        repeating-linear-gradient(to right, #16181a0f 0 1px, transparent 1px 48px),
                        repeating-linear-gradient(to bottom, #16181a0f 0 1px, transparent 1px 48px);
                    color: var(--ink);
                    font-size: 15px;
                    line-height: 1.55;
                }}
                .sheet {{
                    max-width: 1180px;
                    margin: 0 auto;
                    background: var(--sheet);
                    border: 1px solid var(--rule);
                    box-shadow: 0 1px 1px #16181a0f, 0 26px 52px -30px #16181a5c;
                    padding: clamp(20px, 4vw, 48px);
                }}
                .masthead {{
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    gap: 16px 32px;
                    flex-wrap: wrap;
                    border-bottom: 1px solid var(--ink);
                    padding-bottom: 14px;
                    margin-bottom: 32px;
                }}
                .lockup {{ display: block; height: 19px; width: auto; color: var(--ink); }}
                .stamp {{
                    margin: 0;
                    font-family: var(--mono);
                    font-size: 10.5px;
                    letter-spacing: 0.07em;
                    text-transform: uppercase;
                    color: var(--muted);
                    text-align: right;
                }}
                h1 {{
                    font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem);
                    font-weight: 600;
                    letter-spacing: -0.022em;
                    line-height: 1.15;
                    margin: 0 0 10px;
                }}
                .lede {{ margin: 0 0 32px; max-width: 68ch; color: var(--muted); }}
                /* Divisional readout: the run totals on one ruled strip. */
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    border-top: 1px solid var(--ink);
                    border-bottom: 1px solid var(--rule);
                }}
                .division {{
                    position: relative;
                    padding: 15px 16px 15px 15px;
                    border-left: 1px solid var(--rule-soft);
                }}
                .division:first-child {{ border-left: 0; padding-left: 0; }}
                .division::before {{
                    content: "";
                    position: absolute;
                    top: 0; left: -1px;
                    width: 1px; height: 6px;
                    background: var(--ink);
                }}
                .division:first-child::before {{ display: none; }}
                .stat-value {{
                    display: block;
                    font-family: var(--mono);
                    font-size: 1.375rem;
                    font-variant-numeric: tabular-nums;
                    line-height: 1.1;
                    letter-spacing: -0.015em;
                }}
                .stat-label {{
                    display: block;
                    margin-top: 7px;
                    font-size: 10.5px;
                    text-transform: uppercase;
                    letter-spacing: 0.09em;
                    color: var(--muted);
                }}
                .panel {{ margin: 44px 0 0; }}
                .panel h2 {{
                    position: relative;
                    font-size: 1.0625rem;
                    font-weight: 600;
                    letter-spacing: -0.01em;
                    border-top: 1px solid var(--ink);
                    padding-top: 13px;
                    margin: 0 0 16px;
                }}
                .panel h2::before {{
                    content: "";
                    position: absolute;
                    top: 0; left: 0;
                    width: 2px; height: 7px;
                    background: var(--signal);
                }}
                .chart-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                    gap: 12px 32px;
                }}
                footer {{
                    margin-top: 52px;
                    border-top: 1px solid var(--ink);
                    padding-top: 14px;
                    font-family: var(--mono);
                    font-size: 10.5px;
                    letter-spacing: 0.07em;
                    text-transform: uppercase;
                    color: var(--muted);
                }}
            </style>
        </head>
        <body>
        <div class="sheet">
            <header class="masthead">
                {LOCKUP_SVG}
                <p class="stamp">GPU performance dashboard</p>
            </header>
            <h1>Token inference profile</h1>
            <p class="lede">Every panel below is measured from the traced run.
            Latency, bottlenecks, and utilisation are shown side by side so a
            slow token can be attributed to an operation rather than guessed at.</p>

            <div class="stats">
                <div class="division">
                    <span class="stat-value">{len(analyses)}</span>
                    <span class="stat-label">Tokens traced</span>
                </div>
                <div class="division">
                    <span class="stat-value">{total_latency:.1f} ms</span>
                    <span class="stat-label">Total latency</span>
                </div>
                <div class="division">
                    <span class="stat-value">{avg_latency:.1f} ms</span>
                    <span class="stat-label">Mean per token</span>
                </div>
                <div class="division">
                    <span class="stat-value">{avg_performance:.1f}</span>
                    <span class="stat-label">Mean performance score</span>
                </div>
            </div>

            <section class="panel">
                <h2>Token performance timeline</h2>
                {flame_graph}
            </section>

            <div class="chart-grid">
                <section class="panel">
                    <h2>Bottleneck distribution</h2>
                    {bottleneck_dist}
                </section>
                <section class="panel">
                    <h2>Performance trends</h2>
                    {trend_chart}
                </section>
            </div>

            <section class="panel">
                <h2>Operation performance heatmap</h2>
                {heatmap}
            </section>

            <div class="chart-grid">
                <section class="panel">
                    <h2>GPU metrics profile (token 0)</h2>
                    {sample_radar}
                </section>
                <section class="panel">
                    <h2>Operation breakdown (token 0)</h2>
                    {sample_breakdown}
                </section>
            </div>

            <section class="panel">
                <h2>Analysis summary</h2>
                <p>{len(analyses)} token analyses were generated with GPU performance
                metrics attached. Primary bottlenecks and optimization opportunities
                are identified in the panels above.</p>
            </section>

            <footer>Generated by llmtracefx visualize</footer>
        </div>
        <script>
        // Each chart is plotted by its own inline script as the document
        // streams, so a chart inside .chart-grid can be measured before the
        // grid has resolved its track width and then stay drawn at the full
        // content width, overflowing its column. Plotly's responsive handler
        // listens for window resize, so one nudge after load re-fits every
        // chart to the container it actually ended up in.
        window.addEventListener('load', function () {{
            window.dispatchEvent(new Event('resize'));
        }});
        </script>
        </body>
        </html>
        """

        return html_template

    def export_data_json(self, analyses: list[TokenAnalysis]) -> str:
        """Export analysis data as JSON"""
        data = []

        for analysis in analyses:
            ops_data = []
            for op in analysis.operations:
                ops_data.append(
                    {
                        "name": op.name,
                        "duration": op.duration,
                        "start_time": op.start_time,
                    }
                )

            data.append(
                {
                    "token_id": analysis.token_id,
                    "token_text": analysis.token_text,
                    "total_latency_ms": analysis.total_latency_ms,
                    "performance_score": analysis.performance_score,
                    "bottleneck_type": analysis.bottleneck_type,
                    "optimization_flags": analysis.optimization_flags,
                    "operations": ops_data,
                    "gpu_metrics": {
                        "stall_pct": analysis.gpu_metrics.stall_pct,
                        "launch_delay_ms": analysis.gpu_metrics.launch_delay_ms,
                        "memory_latency_ms": analysis.gpu_metrics.memory_latency_ms,
                        "sm_occupancy_pct": analysis.gpu_metrics.sm_occupancy_pct,
                        "cache_hit_rate": analysis.gpu_metrics.cache_hit_rate,
                        "memory_bandwidth_gb_s": analysis.gpu_metrics.memory_bandwidth_gb_s,
                        "compute_utilization": analysis.gpu_metrics.compute_utilization,
                        "occupancy_label": analysis.gpu_metrics.occupancy_label,
                        "metrics_source": analysis.gpu_metrics.metrics_source,
                    },
                }
            )

        return json.dumps(data, indent=2)
