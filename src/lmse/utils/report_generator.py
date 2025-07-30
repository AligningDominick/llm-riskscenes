"""Report generator for evaluation results."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, config: Config):
        """Initialize report generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.template_dir = Path(__file__).parent.parent / "templates"
        
    def generate(
        self,
        results: pd.DataFrame,
        analysis: Dict[str, Any],
        output_path: Path
    ):
        """Generate comprehensive HTML report.
        
        Args:
            results: Evaluation results DataFrame
            analysis: Analysis results dictionary
            output_path: Output file path
        """
        # Generate visualizations
        charts = self._generate_charts(results, analysis)
        
        # Generate HTML report
        html_content = self._generate_html(results, analysis, charts)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_charts(
        self,
        results: pd.DataFrame,
        analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate all charts for the report.
        
        Args:
            results: Results DataFrame
            analysis: Analysis dictionary
            
        Returns:
            Dictionary of chart HTML strings
        """
        charts = {}
        
        # Safety score distribution
        charts["safety_distribution"] = self._create_safety_distribution_chart(results)
        
        # Risk level pie chart
        charts["risk_levels"] = self._create_risk_level_chart(results)
        
        # Language performance
        charts["language_performance"] = self._create_language_performance_chart(results)
        
        # Domain performance
        charts["domain_performance"] = self._create_domain_performance_chart(results)
        
        # Heatmap
        charts["risk_heatmap"] = self._create_risk_heatmap(results)
        
        # Timeline chart if multiple models
        if "model" in results.columns and results["model"].nunique() > 1:
            charts["model_comparison"] = self._create_model_comparison_chart(results)
        
        return charts
    
    def _create_safety_distribution_chart(self, results: pd.DataFrame) -> str:
        """Create safety score distribution chart."""
        fig = px.histogram(
            results,
            x="safety_score",
            nbins=20,
            title="Safety Score Distribution",
            labels={"safety_score": "Safety Score", "count": "Frequency"}
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Safety Score",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        return fig.to_html(div_id="safety_distribution", include_plotlyjs=False)
    
    def _create_risk_level_chart(self, results: pd.DataFrame) -> str:
        """Create risk level pie chart."""
        risk_counts = results["risk_level"].value_counts()
        
        # Define colors for risk levels
        colors = {
            "LOW": "#2ecc71",
            "MEDIUM": "#f39c12",
            "HIGH": "#e74c3c",
            "CRITICAL": "#c0392b"
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[colors.get(level, "#95a5a6") for level in risk_counts.index]),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            template="plotly_white"
        )
        
        return fig.to_html(div_id="risk_levels", include_plotlyjs=False)
    
    def _create_language_performance_chart(self, results: pd.DataFrame) -> str:
        """Create language performance chart."""
        lang_stats = results.groupby("language").agg({
            "safety_score": ["mean", "std", "count"],
            "risk_level": lambda x: (x == "CRITICAL").sum()
        }).round(2)
        
        fig = go.Figure()
        
        # Add bar chart for average safety score
        fig.add_trace(go.Bar(
            x=lang_stats.index,
            y=lang_stats[("safety_score", "mean")],
            name="Average Safety Score",
            error_y=dict(
                type='data',
                array=lang_stats[("safety_score", "std")],
                visible=True
            ),
            marker_color="lightblue"
        ))
        
        fig.update_layout(
            title="Safety Performance by Language",
            xaxis_title="Language",
            yaxis_title="Safety Score",
            template="plotly_white",
            showlegend=False
        )
        
        return fig.to_html(div_id="language_performance", include_plotlyjs=False)
    
    def _create_domain_performance_chart(self, results: pd.DataFrame) -> str:
        """Create domain performance chart."""
        domain_stats = results.groupby("domain").agg({
            "safety_score": "mean",
            "compliance_score": "mean",
            "cultural_score": "mean"
        }).round(2)
        
        fig = go.Figure()
        
        # Add traces for each score type
        for col in domain_stats.columns:
            fig.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=domain_stats.index,
                y=domain_stats[col]
            ))
        
        fig.update_layout(
            title="Performance Scores by Domain",
            xaxis_title="Domain",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white"
        )
        
        return fig.to_html(div_id="domain_performance", include_plotlyjs=False)
    
    def _create_risk_heatmap(self, results: pd.DataFrame) -> str:
        """Create risk heatmap by language and domain."""
        pivot_data = results.pivot_table(
            values="risk_score",
            index="language",
            columns="domain",
            aggfunc="mean"
        ).round(2)
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Domain", y="Language", color="Risk Score"),
            title="Risk Score Heatmap",
            color_continuous_scale="RdYlGn_r",
            aspect="auto"
        )
        
        fig.update_traces(text=pivot_data.values, texttemplate="%{text}")
        fig.update_layout(template="plotly_white")
        
        return fig.to_html(div_id="risk_heatmap", include_plotlyjs=False)
    
    def _create_model_comparison_chart(self, results: pd.DataFrame) -> str:
        """Create model comparison chart."""
        model_stats = results.groupby("model").agg({
            "safety_score": "mean",
            "compliance_score": "mean",
            "cultural_score": "mean"
        }).round(2)
        
        fig = go.Figure()
        
        for col in model_stats.columns:
            fig.add_trace(go.Scatter(
                x=model_stats.index,
                y=model_stats[col],
                mode='lines+markers',
                name=col.replace("_", " ").title(),
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig.to_html(div_id="model_comparison", include_plotlyjs=False)
    
    def _generate_html(
        self,
        results: pd.DataFrame,
        analysis: Dict[str, Any],
        charts: Dict[str, str]
    ) -> str:
        """Generate HTML report content.
        
        Args:
            results: Results DataFrame
            analysis: Analysis dictionary
            charts: Dictionary of chart HTML
            
        Returns:
            Complete HTML content
        """
        summary = analysis["summary"]
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Safety Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .risk-critical {{
            color: #c0392b;
            font-weight: bold;
        }}
        
        .risk-high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .risk-medium {{
            color: #f39c12;
        }}
        
        .risk-low {{
            color: #2ecc71;
        }}
        
        .recommendation {{
            padding: 10px;
            margin: 5px 0;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 3px;
        }}
        
        .chart-container {{
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Safety Evaluation Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="card">
        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Total Evaluations</div>
                <div class="metric-value">{summary['total_evaluations']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Overall Safety Score</div>
                <div class="metric-value">{summary['overall_safety_score']:.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Pass Rate</div>
                <div class="metric-value">{summary['pass_rate']:.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Languages Tested</div>
                <div class="metric-value">{len(summary['languages_tested'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Domains Tested</div>
                <div class="metric-value">{len(summary['domains_tested'])}</div>
            </div>
        </div>
        
        <h3>Risk Summary</h3>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Critical Risks</div>
                <div class="metric-value risk-critical">{summary['critical_failures']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">High Risks</div>
                <div class="metric-value risk-high">{summary['high_risk_count']}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Key Recommendations</h2>
        {self._format_recommendations(analysis['recommendations'])}
    </div>
    
    <div class="card">
        <h2>Risk Distribution</h2>
        <div class="chart-container">
            {charts['risk_levels']}
        </div>
    </div>
    
    <div class="card">
        <h2>Safety Score Distribution</h2>
        <div class="chart-container">
            {charts['safety_distribution']}
        </div>
    </div>
    
    <div class="card">
        <h2>Performance by Language</h2>
        <div class="chart-container">
            {charts['language_performance']}
        </div>
        {self._format_language_table(analysis['by_language'])}
    </div>
    
    <div class="card">
        <h2>Performance by Domain</h2>
        <div class="chart-container">
            {charts['domain_performance']}
        </div>
        {self._format_domain_table(analysis['by_domain'])}
    </div>
    
    <div class="card">
        <h2>Risk Heatmap</h2>
        <div class="chart-container">
            {charts['risk_heatmap']}
        </div>
    </div>
    
    {"<div class='card'><h2>Model Comparison</h2><div class='chart-container'>" + charts.get('model_comparison', '') + "</div></div>" if 'model_comparison' in charts else ""}
    
    <div class="card">
        <h2>Critical Failures</h2>
        {self._format_critical_failures(analysis['critical_failures'][:10])}
    </div>
    
    <div class="footer">
        <p>Generated by LLM Multilingual Safety Evaluation Framework</p>
    </div>
    
    <script>
        // Ensure all Plotly charts are responsive
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize();
        }});
    </script>
</body>
</html>
"""
        return html
    
    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations as HTML."""
        if not recommendations:
            return "<p>No specific recommendations at this time.</p>"
        
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>\n'
        return html
    
    def _format_language_table(self, language_data: dict) -> str:
        """Format language performance table."""
        html = """
        <table>
            <thead>
                <tr>
                    <th>Language</th>
                    <th>Evaluations</th>
                    <th>Avg Safety Score</th>
                    <th>Avg Compliance</th>
                    <th>Critical Failures</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for lang, data in language_data.items():
            html += f"""
                <tr>
                    <td>{lang.title()}</td>
                    <td>{data['count']}</td>
                    <td>{data['avg_safety_score']:.1f}</td>
                    <td>{data['avg_compliance_score']:.1f}</td>
                    <td class="risk-critical">{data['critical_failures']}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html
    
    def _format_domain_table(self, domain_data: dict) -> str:
        """Format domain performance table."""
        html = """
        <table>
            <thead>
                <tr>
                    <th>Domain</th>
                    <th>Evaluations</th>
                    <th>Avg Safety Score</th>
                    <th>Avg Compliance</th>
                    <th>Top Risk</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for domain, data in domain_data.items():
            top_risk = data['top_risks'][0]['issue'] if data['top_risks'] else "None"
            html += f"""
                <tr>
                    <td>{domain.title()}</td>
                    <td>{data['count']}</td>
                    <td>{data['avg_safety_score']:.1f}</td>
                    <td>{data['avg_compliance_score']:.1f}</td>
                    <td>{top_risk}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html
    
    def _format_critical_failures(self, failures: list) -> str:
        """Format critical failures table."""
        if not failures:
            return "<p>No critical failures detected.</p>"
        
        html = """
        <table>
            <thead>
                <tr>
                    <th>Scenario ID</th>
                    <th>Language</th>
                    <th>Domain</th>
                    <th>Risk Score</th>
                    <th>Main Issue</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for failure in failures:
            main_issue = failure['issues'][0] if failure['issues'] else "Unknown"
            html += f"""
                <tr>
                    <td>{failure['scenario_id']}</td>
                    <td>{failure['language']}</td>
                    <td>{failure['domain']}</td>
                    <td class="risk-critical">{failure['risk_score']}</td>
                    <td>{main_issue}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        return html
    
    def generate_analysis_report(self, analysis: dict, output_path: Path):
        """Generate a standalone analysis report.
        
        Args:
            analysis: Analysis dictionary
            output_path: Output file path
        """
        # For now, create a simple HTML report
        # This could be extended with more sophisticated templates
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>LLM Safety Evaluation Analysis</h1>
    <h2>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
    <pre>{json.dumps(analysis, indent=2)}</pre>
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)