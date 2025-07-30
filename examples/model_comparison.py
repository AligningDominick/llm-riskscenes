"""
Model comparison example for LLM Multilingual Safety Evaluation Framework.

This example demonstrates how to:
1. Evaluate multiple models
2. Compare their performance
3. Generate comparative analysis
"""

import asyncio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lmse import SafetyEvaluator, ModelLoader, Config


async def evaluate_model(model_name: str, evaluator: SafetyEvaluator, scenarios: list) -> pd.DataFrame:
    """Evaluate a single model."""
    print(f"\nEvaluating {model_name}...")
    
    # Load model (using mock for example)
    model = ModelLoader.create_mock_model(
        name=model_name,
        responses={
            "medical": f"[{model_name}] I cannot provide medical advice. Please consult a healthcare professional.",
            "legal": f"[{model_name}] For legal matters, please consult with a qualified attorney.",
            "investment": f"[{model_name}] I cannot provide investment advice. Please consult a financial advisor.",
        },
        default_response=f"[{model_name}] I'll provide general information while prioritizing safety."
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        model=model,
        scenarios=scenarios,
        async_batch_size=10
    )
    
    return results


def compare_models(all_results: dict):
    """Generate comparative analysis of models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    # Combine all results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # Overall comparison
    print("\nOVERALL PERFORMANCE:")
    print("-"*40)
    
    model_summary = combined_df.groupby('model').agg({
        'safety_score': ['mean', 'std'],
        'compliance_score': 'mean',
        'cultural_score': 'mean',
        'risk_level': lambda x: (x == 'CRITICAL').sum()
    }).round(2)
    
    print(model_summary)
    
    # Performance by language
    print("\n\nPERFORMANCE BY LANGUAGE:")
    print("-"*40)
    
    lang_comparison = combined_df.pivot_table(
        values='safety_score',
        index='language',
        columns='model',
        aggfunc='mean'
    ).round(2)
    
    print(lang_comparison)
    
    # Performance by domain
    print("\n\nPERFORMANCE BY DOMAIN:")
    print("-"*40)
    
    domain_comparison = combined_df.pivot_table(
        values='safety_score',
        index='domain',
        columns='model',
        aggfunc='mean'
    ).round(2)
    
    print(domain_comparison)
    
    # Critical failures comparison
    print("\n\nCRITICAL FAILURES BY MODEL:")
    print("-"*40)
    
    critical_counts = combined_df[combined_df['risk_level'] == 'CRITICAL'].groupby('model').size()
    print(critical_counts)
    
    return combined_df, model_summary, lang_comparison, domain_comparison


def create_comparison_plots(combined_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for model comparison."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall safety scores by model
    plt.figure(figsize=(10, 6))
    model_scores = combined_df.groupby('model')['safety_score'].agg(['mean', 'std'])
    
    plt.bar(model_scores.index, model_scores['mean'], yerr=model_scores['std'], capsize=10)
    plt.xlabel('Model')
    plt.ylabel('Safety Score')
    plt.title('Average Safety Score by Model')
    plt.ylim(0, 100)
    plt.savefig(output_dir / 'model_safety_scores.png')
    plt.close()
    
    # 2. Risk level distribution by model
    fig, axes = plt.subplots(1, len(combined_df['model'].unique()), figsize=(15, 5))
    
    for idx, (model, group) in enumerate(combined_df.groupby('model')):
        risk_counts = group['risk_level'].value_counts()
        colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
        
        ax = axes[idx] if len(axes) > 1 else axes
        risk_counts.plot(kind='pie', ax=ax, colors=[colors.get(x, 'gray') for x in risk_counts.index])
        ax.set_title(f'{model} Risk Distribution')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distribution_by_model.png')
    plt.close()
    
    # 3. Language performance heatmap
    plt.figure(figsize=(12, 8))
    lang_pivot = combined_df.pivot_table(
        values='safety_score',
        index='language',
        columns='model',
        aggfunc='mean'
    )
    
    sns.heatmap(lang_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=70)
    plt.title('Safety Score Heatmap: Language vs Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'language_model_heatmap.png')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def main():
    # Configuration
    config = Config.default()
    config.set("evaluation.sample_size", 20)  # Small sample for demo
    config.set("output.directory", "results/model_comparison")
    
    # Models to compare
    models_to_test = [
        "model-a",
        "model-b", 
        "model-c"
    ]
    
    # Initialize evaluator
    evaluator = SafetyEvaluator(config)
    
    # Load scenarios once
    from lmse.data.scenario_loader import ScenarioLoader
    loader = ScenarioLoader(config)
    scenarios = loader.load_scenarios(
        languages=["english", "spanish", "chinese"],
        domains=["healthcare", "legal", "finance"],
        sample_size=20
    )
    
    print(f"Loaded {len(scenarios)} scenarios for evaluation")
    
    # Evaluate each model
    all_results = {}
    
    # Run evaluations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    for model_name in models_to_test:
        results = loop.run_until_complete(
            evaluate_model(model_name, evaluator, scenarios)
        )
        all_results[model_name] = results
    
    # Compare models
    combined_df, model_summary, lang_comparison, domain_comparison = compare_models(all_results)
    
    # Create visualizations
    output_dir = Path(config.get("output.directory"))
    create_comparison_plots(combined_df, output_dir)
    
    # Save comparison results
    comparison_file = output_dir / "model_comparison_results.xlsx"
    with pd.ExcelWriter(comparison_file) as writer:
        model_summary.to_excel(writer, sheet_name='Overall Summary')
        lang_comparison.to_excel(writer, sheet_name='Language Comparison')
        domain_comparison.to_excel(writer, sheet_name='Domain Comparison')
        combined_df.to_excel(writer, sheet_name='All Results', index=False)
    
    print(f"\nComparison results saved to {comparison_file}")
    
    # Generate recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    best_overall = model_summary[('safety_score', 'mean')].idxmax()
    print(f"\n1. Best Overall Performance: {best_overall}")
    print(f"   - Safety Score: {model_summary.loc[best_overall, ('safety_score', 'mean')]:.2f}")
    
    # Find model with least critical failures
    critical_col = ('risk_level', '<lambda>')
    if critical_col in model_summary.columns:
        least_critical = model_summary[critical_col].idxmin()
        print(f"\n2. Least Critical Failures: {least_critical}")
        print(f"   - Critical Count: {int(model_summary.loc[least_critical, critical_col])}")
    
    # Language-specific recommendations
    print("\n3. Language-Specific Recommendations:")
    for lang in lang_comparison.index:
        best_for_lang = lang_comparison.loc[lang].idxmax()
        print(f"   - {lang}: Use {best_for_lang} (Score: {lang_comparison.loc[lang, best_for_lang]:.2f})")
    
    # Domain-specific recommendations
    print("\n4. Domain-Specific Recommendations:")
    for domain in domain_comparison.index:
        best_for_domain = domain_comparison.loc[domain].idxmax()
        print(f"   - {domain}: Use {best_for_domain} (Score: {domain_comparison.loc[domain, best_for_domain]:.2f})")
    
    print("\nModel comparison complete!")


if __name__ == "__main__":
    main()