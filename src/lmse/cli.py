"""Command-line interface for the evaluation framework."""

import click
import json
import yaml
from pathlib import Path
from typing import Optional, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

from . import __version__
from .core.evaluator import SafetyEvaluator
from .models.loader import ModelLoader
from .utils.config import Config
from .data.scenario_loader import ScenarioLoader


console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """LLM Multilingual Safety Evaluation Framework CLI."""
    pass


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Configuration file path"
)
@click.option(
    "--model", "-m",
    type=str,
    required=True,
    help="Model name to evaluate"
)
@click.option(
    "--languages", "-l",
    multiple=True,
    help="Languages to evaluate (can specify multiple)"
)
@click.option(
    "--domains", "-d",
    multiple=True,
    help="Domains to evaluate (can specify multiple)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for results"
)
@click.option(
    "--sample-size", "-s",
    type=int,
    help="Number of scenarios to sample"
)
@click.option(
    "--async/--sync",
    default=True,
    help="Use async evaluation (default: async)"
)
def evaluate(
    config: Optional[str],
    model: str,
    languages: List[str],
    domains: List[str],
    output: Optional[str],
    sample_size: Optional[int],
    async: bool
):
    """Run safety evaluation on a model."""
    console.print(f"[bold blue]LLM Safety Evaluation v{__version__}[/bold blue]")
    
    # Load configuration
    if config:
        cfg = Config.load(config)
    else:
        cfg = Config.default()
    
    # Override with CLI options
    if output:
        cfg.set("output.directory", output)
    cfg.set("evaluation.async", async)
    
    # Load model
    console.print(f"[yellow]Loading model: {model}[/yellow]")
    try:
        model_instance = ModelLoader.load(model, cfg)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return
    
    # Initialize evaluator
    evaluator = SafetyEvaluator(cfg)
    
    # Run evaluation
    console.print("[green]Starting evaluation...[/green]")
    results = evaluator.evaluate(
        model=model_instance,
        languages=list(languages) if languages else None,
        domains=list(domains) if domains else None,
        async_batch_size=cfg.get("evaluation.batch_size", 10)
    )
    
    # Display results summary
    display_results_summary(results)
    
    # Generate report
    report_path = evaluator.generate_report()
    console.print(f"[green]Report generated: {report_path}[/green]")


@cli.command()
@click.option(
    "--results", "-r",
    type=click.Path(exists=True),
    required=True,
    help="Results file to analyze"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output path for analysis report"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "html", "markdown"]),
    default="html",
    help="Output format"
)
def analyze(results: str, output: Optional[str], format: str):
    """Analyze evaluation results."""
    console.print("[bold blue]Analyzing Results[/bold blue]")
    
    # Load results
    results_df = pd.read_csv(results) if results.endswith('.csv') else pd.read_json(results)
    
    # Create evaluator for analysis
    evaluator = SafetyEvaluator(Config.default())
    evaluator.results = results_df.to_dict('records')
    
    # Perform analysis
    analysis = evaluator.analyze(results_df)
    
    # Display analysis
    display_analysis(analysis)
    
    # Save analysis
    if output:
        save_analysis(analysis, output, format)
        console.print(f"[green]Analysis saved to: {output}[/green]")


@cli.command()
@click.option(
    "--languages", "-l",
    multiple=True,
    help="Filter by languages"
)
@click.option(
    "--domains", "-d",
    multiple=True,
    help="Filter by domains"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format"
)
def list_scenarios(languages: List[str], domains: List[str], format: str):
    """List available evaluation scenarios."""
    loader = ScenarioLoader(Config.default())
    scenarios = loader.load_scenarios(
        languages=list(languages) if languages else None,
        domains=list(domains) if domains else None
    )
    
    if format == "table":
        display_scenarios_table(scenarios)
    elif format == "json":
        console.print(json.dumps(scenarios, indent=2))
    elif format == "yaml":
        console.print(yaml.dump(scenarios))


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="configs/custom.yaml",
    help="Output path for configuration"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Configuration format"
)
def init_config(output: str, format: str):
    """Initialize a configuration file."""
    config = Config.default()
    
    # Add example API keys section
    config.set("api_keys", {
        "anthropic": "your-anthropic-api-key",
        "openai": "your-openai-api-key",
        "google": "your-google-api-key"
    })
    
    # Save configuration
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "yaml":
        with open(output_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
    else:
        with open(output_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    console.print(f"[green]Configuration created: {output_path}[/green]")


@cli.command()
def list_models():
    """List available models."""
    models = ModelLoader.list_available_models()
    
    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Type", style="green")
    
    for name, description in models.items():
        table.add_row(name, description)
    
    console.print(table)


@cli.command()
@click.argument("scenario_file", type=click.Path(exists=True))
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate scenarios before adding"
)
def add_scenarios(scenario_file: str, validate: bool):
    """Add scenarios from a file."""
    loader = ScenarioLoader(Config.default())
    
    # Load scenarios from file
    if scenario_file.endswith('.json'):
        with open(scenario_file, 'r') as f:
            new_scenarios = json.load(f)
    else:
        with open(scenario_file, 'r') as f:
            new_scenarios = yaml.safe_load(f)
    
    if not isinstance(new_scenarios, list):
        new_scenarios = [new_scenarios]
    
    # Add scenarios
    added = 0
    for scenario in new_scenarios:
        try:
            loader.add_scenario(scenario)
            added += 1
        except Exception as e:
            console.print(f"[red]Error adding scenario: {e}[/red]")
    
    console.print(f"[green]Added {added} scenarios[/green]")


def display_results_summary(results: pd.DataFrame):
    """Display evaluation results summary."""
    table = Table(title="Evaluation Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Calculate metrics
    total = len(results)
    avg_safety = results["safety_score"].mean()
    critical = len(results[results["risk_level"] == "CRITICAL"])
    high = len(results[results["risk_level"] == "HIGH"])
    
    table.add_row("Total Scenarios", str(total))
    table.add_row("Average Safety Score", f"{avg_safety:.2f}")
    table.add_row("Critical Risks", str(critical))
    table.add_row("High Risks", str(high))
    
    console.print(table)
    
    # Language breakdown
    lang_table = Table(title="Results by Language")
    lang_table.add_column("Language", style="cyan")
    lang_table.add_column("Count", style="green")
    lang_table.add_column("Avg Safety", style="yellow")
    
    for lang in results["language"].unique():
        lang_results = results[results["language"] == lang]
        lang_table.add_row(
            lang,
            str(len(lang_results)),
            f"{lang_results['safety_score'].mean():.2f}"
        )
    
    console.print(lang_table)


def display_analysis(analysis: dict):
    """Display analysis results."""
    # Summary
    summary = analysis["summary"]
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"Total Evaluations: {summary['total_evaluations']}")
    console.print(f"Overall Safety Score: {summary['overall_safety_score']:.2f}")
    console.print(f"Pass Rate: {summary['pass_rate']:.2%}")
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    for rec in analysis["recommendations"]:
        console.print(f"• {rec}")
    
    # Critical failures
    if analysis["critical_failures"]:
        console.print("\n[bold red]Critical Failures:[/bold red]")
        for failure in analysis["critical_failures"][:5]:
            console.print(f"• {failure['scenario_id']} ({failure['language']}/{failure['domain']})")


def display_scenarios_table(scenarios: List[dict]):
    """Display scenarios in a table."""
    table = Table(title="Available Scenarios")
    table.add_column("ID", style="cyan")
    table.add_column("Language", style="green")
    table.add_column("Domain", style="yellow")
    table.add_column("Tags", style="magenta")
    
    for scenario in scenarios:
        tags = ", ".join(scenario.get("tags", []))
        table.add_row(
            scenario["id"],
            scenario["language"],
            scenario["domain"],
            tags
        )
    
    console.print(table)


def save_analysis(analysis: dict, output: str, format: str):
    """Save analysis to file."""
    if format == "json":
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
    elif format == "html":
        # Generate HTML report
        from ..utils.report_generator import ReportGenerator
        generator = ReportGenerator(Config.default())
        generator.generate_analysis_report(analysis, Path(output))
    elif format == "markdown":
        # Generate markdown report
        with open(output, 'w') as f:
            f.write("# LLM Safety Evaluation Analysis\n\n")
            f.write("## Summary\n")
            for key, value in analysis["summary"].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n## Recommendations\n")
            for rec in analysis["recommendations"]:
                f.write(f"- {rec}\n")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()