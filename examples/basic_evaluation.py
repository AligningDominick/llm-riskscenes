"""
Basic evaluation example for LLM Multilingual Safety Evaluation Framework.

This example demonstrates how to:
1. Load a model
2. Run safety evaluation
3. Analyze results
4. Generate reports
"""

import os
from pathlib import Path
from lmse import SafetyEvaluator, ModelLoader, Config

def main():
    # Set up configuration
    config = Config.default()
    
    # Override some settings for this example
    config.set("evaluation.sample_size", 10)  # Use only 10 scenarios for quick demo
    config.set("output.directory", "results/example")
    
    # Initialize model (using mock model for example)
    # In production, replace with actual model like 'claude-3-opus' or 'gpt-4'
    print("Loading model...")
    model = ModelLoader.create_mock_model(
        name="demo-model",
        responses={
            # Add some example responses for specific prompts
            "medication": "I cannot provide specific medication dosages. Please consult a healthcare professional.",
            "legal": "I'm not qualified to provide legal advice. Please consult with a licensed attorney.",
            "investment": "I cannot provide specific investment advice. Consider consulting a financial advisor.",
        },
        default_response="I'll do my best to provide helpful information while prioritizing safety."
    )
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = SafetyEvaluator(config)
    
    # Run evaluation on specific languages and domains
    print("\nRunning evaluation...")
    results = evaluator.evaluate(
        model=model,
        languages=["english", "spanish", "chinese"],
        domains=["healthcare", "legal", "finance"],
        async_batch_size=5
    )
    
    print(f"\nCompleted {len(results)} evaluations")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = evaluator.analyze(results)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary = analysis["summary"]
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Overall Safety Score: {summary['overall_safety_score']:.2f}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Critical Failures: {summary['critical_failures']}")
    print(f"High Risk Count: {summary['high_risk_count']}")
    
    # Print language breakdown
    print("\n" + "-"*50)
    print("PERFORMANCE BY LANGUAGE")
    print("-"*50)
    
    for lang, stats in analysis["by_language"].items():
        print(f"\n{lang.upper()}:")
        print(f"  - Average Safety Score: {stats['avg_safety_score']:.2f}")
        print(f"  - Critical Failures: {stats['critical_failures']}")
        print(f"  - Evaluations: {stats['count']}")
    
    # Print domain breakdown
    print("\n" + "-"*50)
    print("PERFORMANCE BY DOMAIN")
    print("-"*50)
    
    for domain, stats in analysis["by_domain"].items():
        print(f"\n{domain.upper()}:")
        print(f"  - Average Safety Score: {stats['avg_safety_score']:.2f}")
        print(f"  - Average Compliance: {stats['avg_compliance_score']:.2f}")
        if stats['top_risks']:
            print(f"  - Top Risk: {stats['top_risks'][0]['issue']}")
    
    # Print recommendations
    print("\n" + "-"*50)
    print("RECOMMENDATIONS")
    print("-"*50)
    
    for i, rec in enumerate(analysis["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Generate report
    print("\n" + "="*50)
    print("Generating HTML report...")
    report_path = evaluator.generate_report()
    print(f"Report saved to: {report_path}")
    
    # Save raw results
    results_path = Path(config.get("output.directory")) / "evaluation_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Raw results saved to: {results_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()