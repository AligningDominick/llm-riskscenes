# LLM Multilingual Safety Evaluation Framework

A comprehensive framework for evaluating Large Language Model (LLM) safety across multiple languages and cultural contexts.

## 🌍 Overview

This framework provides a systematic approach to assess LLM behavior in safety-critical scenarios across different languages, domains, and cultural contexts. It helps identify potential risks and biases in multilingual AI systems.

## ✨ Features

- **Multi-Model Support**: Evaluate Claude, GPT, Gemini, and other LLMs
- **Extensive Language Coverage**: 15+ languages including low-resource languages
- **Domain-Specific Scenarios**: Healthcare, Education, Legal, Finance, and more
- **Advanced Risk Analysis**: Comprehensive scoring and risk assessment
- **Interactive Visualizations**: Rich dashboards and reports
- **Configurable Pipeline**: YAML-based configuration system
- **CLI Interface**: Easy-to-use command-line tools
- **Extensible Architecture**: Easy to add new models, languages, and scenarios

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llm-multilingual-safety-eval.git
cd llm-multilingual-safety-eval

# Install dependencies
pip install -e .
```

### Basic Usage

```bash
# Run evaluation with default settings
python -m lmse evaluate --config configs/default.yaml

# Evaluate specific model on specific language
python -m lmse evaluate --model claude-3 --language hindi --domain healthcare

# Generate risk report
python -m lmse analyze --input results/latest --output reports/
```

### Python API

```python
from lmse import SafetyEvaluator, ModelLoader

# Initialize evaluator
evaluator = SafetyEvaluator(config_path="configs/default.yaml")

# Load model
model = ModelLoader.load("claude-3", api_key="your-key")

# Run evaluation
results = evaluator.evaluate(
    model=model,
    languages=["hindi", "arabic", "swahili"],
    domains=["healthcare", "education"]
)

# Analyze results
analysis = evaluator.analyze(results)
print(analysis.summary())
```

## 📊 Architecture

```
lmse/
├── src/
│   ├── core/           # Core evaluation engine
│   ├── models/         # Model interfaces and wrappers
│   ├── evaluators/     # Evaluation logic
│   ├── analyzers/      # Analysis and scoring
│   ├── utils/          # Utilities and helpers
│   └── data/           # Data loaders and processors
├── datasets/           # Multilingual scenarios
├── configs/            # Configuration files
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── examples/           # Example usage scripts
```

## 🌐 Supported Languages

- **High-Resource**: English, Chinese, Spanish, French, German, Japanese
- **Medium-Resource**: Hindi, Arabic, Portuguese, Russian, Korean
- **Low-Resource**: Swahili, Yoruba, Bengali, Vietnamese, Thai

## 🏥 Evaluation Domains

- **Healthcare**: Medical advice, diagnosis, treatment recommendations
- **Education**: Learning guidance, academic integrity, child safety
- **Legal**: Legal advice, rights information, compliance
- **Finance**: Financial advice, investment, fraud detection
- **Civic**: Political information, voting, civic engagement
- **Social**: Social interactions, cultural sensitivity, bias

## 📈 Metrics and Scoring

- **Safety Score**: Overall safety rating (0-100)
- **Risk Level**: Critical, High, Medium, Low
- **Compliance**: Adherence to expected behaviors
- **Cultural Sensitivity**: Context-appropriate responses
- **Bias Detection**: Identification of linguistic or cultural biases

## 🔧 Configuration

```yaml
# configs/default.yaml
evaluation:
  models:
    - claude-3-opus
    - gpt-4
  languages:
    - hindi
    - arabic
    - swahili
  domains:
    - healthcare
    - education
  
scoring:
  weights:
    safety: 0.4
    compliance: 0.3
    cultural_sensitivity: 0.3
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api_reference.md)
- [Contributing Guidelines](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and researchers in AI safety
- Special thanks to linguistic experts who helped create culturally appropriate scenarios

## 📧 Contact

For questions or collaboration: dominickaligning@gmail.com

---

**Note**: This framework is for research and evaluation purposes. Always consult appropriate professionals for domain-specific advice.
