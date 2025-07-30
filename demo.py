"""
Simple demonstration of the LLM Multilingual Safety Evaluation Framework.
This script shows how to quickly evaluate a model's safety.
"""

from lmse import SafetyEvaluator, ModelLoader, Config

def main():
    print("🚀 LLM Multilingual Safety Evaluation - Quick Demo")
    print("=" * 50)
    
    # 1. Setup configuration
    config = Config.default()
    config.set("evaluation.sample_size", 5)  # Use only 5 scenarios for demo
    config.set("output.directory", "demo_results")
    
    # 2. Load a mock model for demonstration
    print("\n📦 Loading mock model...")
    model = ModelLoader.create_mock_model(
        name="demo-llm",
        responses={
            "medication": "I cannot provide specific medication advice. Please consult a healthcare professional.",
            "legal": "For legal matters, you should consult with a qualified attorney.",
            "investment": "I cannot provide investment advice. Please speak with a financial advisor.",
            "medical": "I'm not qualified to provide medical diagnosis. Please see a doctor.",
        },
        default_response="I'll do my best to provide helpful general information."
    )
    print(f"✅ Model loaded: {model.name}")
    
    # 3. Initialize evaluator
    print("\n🔧 Initializing evaluator...")
    evaluator = SafetyEvaluator(config)
    
    # 4. Run evaluation on key risk areas
    print("\n🏃 Running safety evaluation...")
    print("   Languages: English, Spanish, Chinese")
    print("   Domains: Healthcare, Legal, Finance")
    
    results = evaluator.evaluate(
        model=model,
        languages=["english", "spanish", "chinese"],
        domains=["healthcare", "legal", "finance"]
    )
    
    print(f"\n✅ Evaluation complete! Evaluated {len(results)} scenarios")
    
    # 5. Analyze results
    print("\n📊 Analyzing results...")
    analysis = evaluator.analyze(results)
    
    # 6. Display key findings
    summary = analysis['summary']
    print("\n" + "="*50)
    print("📈 EVALUATION SUMMARY")
    print("="*50)
    print(f"   Overall Safety Score: {summary['overall_safety_score']:.1f}/100")
    print(f"   Pass Rate: {summary['pass_rate']:.1%}")
    print(f"   Critical Failures: {summary['critical_failures']}")
    print(f"   High Risk Scenarios: {summary['high_risk_count']}")
    
    # 7. Show performance by language
    print("\n📍 Performance by Language:")
    for lang, stats in analysis['by_language'].items():
        print(f"   {lang.capitalize()}: {stats['avg_safety_score']:.1f}/100")
    
    # 8. Show performance by domain
    print("\n🏛️ Performance by Domain:")
    for domain, stats in analysis['by_domain'].items():
        print(f"   {domain.capitalize()}: {stats['avg_safety_score']:.1f}/100")
    
    # 9. Display recommendations
    print("\n💡 Key Recommendations:")
    for i, rec in enumerate(analysis['recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # 10. Generate report
    print("\n📄 Generating detailed report...")
    report_path = evaluator.generate_report()
    print(f"✅ Report saved to: {report_path}")
    
    print("\n" + "="*50)
    print("🎉 Demo complete! Check the report for detailed analysis.")
    print("="*50)


if __name__ == "__main__":
    main()