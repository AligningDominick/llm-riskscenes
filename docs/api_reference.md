# API Reference

## Core Classes

### SafetyEvaluator

The main class for conducting safety evaluations.

```python
from lmse import SafetyEvaluator

evaluator = SafetyEvaluator(config)
```

#### Methods

##### `__init__(config: Union[str, Path, Dict, Config])`

Initialize the safety evaluator.

**Parameters:**
- `config`: Configuration file path, dictionary, or Config object

**Example:**
```python
# From file
evaluator = SafetyEvaluator("configs/default.yaml")

# From dictionary
evaluator = SafetyEvaluator({
    "evaluation": {"async": True},
    "model": {"temperature": 0.7}
})

# From Config object
config = Config.default()
evaluator = SafetyEvaluator(config)
```

##### `evaluate(model: BaseModel, languages: Optional[List[str]] = None, domains: Optional[List[str]] = None, scenarios: Optional[List[Dict]] = None, async_batch_size: int = 10) -> pd.DataFrame`

Run safety evaluation on a model.

**Parameters:**
- `model`: The model instance to evaluate
- `languages`: List of languages to test (None for all)
- `domains`: List of domains to test (None for all)
- `scenarios`: Custom scenarios to test (optional)
- `async_batch_size`: Number of concurrent requests

**Returns:**
- DataFrame with evaluation results

**Example:**
```python
results = evaluator.evaluate(
    model=model,
    languages=["hindi", "arabic"],
    domains=["healthcare", "education"],
    async_batch_size=20
)
```

##### `analyze(results: Optional[pd.DataFrame] = None) -> Dict[str, Any]`

Perform comprehensive analysis on evaluation results.

**Parameters:**
- `results`: DataFrame of results (uses self.results if None)

**Returns:**
- Dictionary containing analysis results

**Example:**
```python
analysis = evaluator.analyze()
print(analysis["summary"]["overall_safety_score"])
print(analysis["recommendations"])
```

##### `generate_report(output_path: Optional[Path] = None) -> Path`

Generate a comprehensive evaluation report.

**Parameters:**
- `output_path`: Path to save the report (auto-generated if None)

**Returns:**
- Path to the generated report

### ModelLoader

Factory class for loading different model implementations.

```python
from lmse import ModelLoader

model = ModelLoader.load("claude-3-opus", api_key="your-key")
```

#### Class Methods

##### `load(model_name: str, config: Optional[Union[Dict, Config, str, Path]] = None, **kwargs) -> BaseModel`

Load a model by name.

**Parameters:**
- `model_name`: Name of the model to load
- `config`: Model configuration
- `**kwargs`: Additional configuration parameters

**Returns:**
- Instantiated model object

**Supported Models:**
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- `gpt-4`, `gpt-3.5-turbo`
- `gemini-pro`, `gemini-ultra`
- `llama-2-70b`, `mistral-large`
- `mock` (for testing)

**Example:**
```python
# Load with API key
model = ModelLoader.load(
    "claude-3-opus",
    api_key="your-anthropic-key"
)

# Load with config file
model = ModelLoader.load(
    "gpt-4",
    config="configs/openai.yaml"
)

# Create mock model for testing
mock_model = ModelLoader.create_mock_model(
    responses={"test": "Mock response"}
)
```

##### `list_available_models() -> Dict[str, str]`

List all available models.

**Returns:**
- Dictionary mapping model names to descriptions

##### `register_model(name: str, model_class: Union[Type[BaseModel], str])`

Register a new model implementation.

**Parameters:**
- `name`: Model identifier
- `model_class`: Model class or import path

### RiskAnalyzer

Analyzer for assessing risks in model responses.

```python
from lmse.analyzers import RiskAnalyzer

analyzer = RiskAnalyzer(config)
```

#### Methods

##### `analyze_response(scenario: Dict[str, Any], response: str) -> Dict[str, Any]`

Analyze a model response for risks.

**Parameters:**
- `scenario`: Scenario information
- `response`: Model response

**Returns:**
- Dictionary containing analysis results with keys:
  - `risk_score`: Overall risk score (0-100)
  - `risk_level`: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
  - `safety_score`: Safety compliance score (0-100)
  - `compliance_score`: Expected behavior compliance (0-100)
  - `cultural_score`: Cultural sensitivity score (0-100)
  - `identified_issues`: List of identified problems
  - `recommendations`: List of improvement recommendations

### Config

Configuration management class.

```python
from lmse.utils import Config

config = Config.load("configs/default.yaml")
```

#### Class Methods

##### `load(source: Union[str, Path, Dict]) -> Config`

Load configuration from various sources.

**Parameters:**
- `source`: Configuration source (file path, dict, or Config object)

##### `default() -> Config`

Get default configuration.

#### Instance Methods

##### `get(key: str, default: Any = None) -> Any`

Get configuration value.

**Parameters:**
- `key`: Configuration key (supports dot notation)
- `default`: Default value if key not found

**Example:**
```python
# Get nested values
batch_size = config.get("evaluation.batch_size", 10)
api_key = config.get("api_keys.anthropic")
```

##### `set(key: str, value: Any)`

Set configuration value.

**Parameters:**
- `key`: Configuration key (supports dot notation)
- `value`: Value to set

### ScenarioLoader

Loader for evaluation scenarios.

```python
from lmse.data import ScenarioLoader

loader = ScenarioLoader(config)
scenarios = loader.load_scenarios()
```

#### Methods

##### `load_scenarios(languages: Optional[List[str]] = None, domains: Optional[List[str]] = None, scenario_ids: Optional[List[str]] = None, sample_size: Optional[int] = None, seed: Optional[int] = None) -> List[Dict]`

Load scenarios based on filters.

**Parameters:**
- `languages`: List of languages to include
- `domains`: List of domains to include
- `scenario_ids`: Specific scenario IDs to load
- `sample_size`: Number of scenarios to sample
- `seed`: Random seed for sampling

**Returns:**
- List of scenario dictionaries

##### `add_scenario(scenario: Dict)`

Add a new scenario to the loader.

##### `get_statistics() -> Dict`

Get statistics about loaded scenarios.

## Base Classes

### BaseModel

Abstract base class for all LLM model implementations.

```python
from lmse.models import BaseModel

class CustomModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

#### Abstract Methods

##### `generate(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> str`

Generate a response for the given prompt.

##### `validate_config() -> bool`

Validate the model configuration.

#### Optional Methods

##### `generate_async(prompt: str, **kwargs) -> str`

Async version of generate method.

## Utility Functions

### Logging

```python
from lmse.utils import setup_logger

logger = setup_logger(
    name="my_module",
    level="INFO",
    log_file="logs/my_module.log"
)
```

### Progress Tracking

```python
from lmse.utils import ProgressLogger

progress = ProgressLogger(total=100)
for i in range(100):
    # Do work
    progress.update(1)
progress.complete()
```

## CLI Commands

### evaluate

Run safety evaluation on a model.

```bash
lmse evaluate [OPTIONS]
```

**Options:**
- `--config, -c`: Configuration file path
- `--model, -m`: Model name to evaluate (required)
- `--languages, -l`: Languages to evaluate (multiple)
- `--domains, -d`: Domains to evaluate (multiple)
- `--output, -o`: Output directory for results
- `--sample-size, -s`: Number of scenarios to sample
- `--async/--sync`: Use async evaluation (default: async)

### analyze

Analyze evaluation results.

```bash
lmse analyze [OPTIONS]
```

**Options:**
- `--results, -r`: Results file to analyze (required)
- `--output, -o`: Output path for analysis report
- `--format, -f`: Output format (json, html, markdown)

### list-scenarios

List available evaluation scenarios.

```bash
lmse list-scenarios [OPTIONS]
```

**Options:**
- `--languages, -l`: Filter by languages (multiple)
- `--domains, -d`: Filter by domains (multiple)
- `--format, -f`: Output format (table, json, yaml)

### init-config

Initialize a configuration file.

```bash
lmse init-config [OPTIONS]
```

**Options:**
- `--output, -o`: Output path for configuration
- `--format, -f`: Configuration format (yaml, json)

### list-models

List available models.

```bash
lmse list-models
```

## Scenario Format

Scenarios are defined as dictionaries with the following structure:

```python
{
    "id": "unique_scenario_id",
    "language": "language_code",
    "domain": "domain_name",
    "prompt": "The prompt text in the target language",
    "expected_behavior": "Description of expected model behavior",
    "system_prompt": "Optional system prompt",
    "tags": ["tag1", "tag2"],
    "risk_factors": ["risk1", "risk2"],
    "cultural_context": {
        "requires_acknowledgment": true,
        "avoid_assumptions": ["assumption1"],
        "cultural_notes": "Additional context"
    },
    "metadata": {
        "source": "scenario_source",
        "version": "1.0",
        "created": "2024-01-01"
    }
}
```

## Response Format

Evaluation results are returned as a pandas DataFrame with the following columns:

- `session_id`: Unique evaluation session identifier
- `timestamp`: Evaluation timestamp
- `model`: Model name
- `scenario_id`: Scenario identifier
- `language`: Scenario language
- `domain`: Scenario domain
- `prompt`: Original prompt
- `expected_behavior`: Expected behavior description
- `response`: Model's response
- `risk_score`: Overall risk score (0-100)
- `risk_level`: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
- `safety_score`: Safety compliance score (0-100)
- `compliance_score`: Expected behavior compliance (0-100)
- `cultural_score`: Cultural sensitivity score (0-100)
- `identified_issues`: List of identified issues
- `recommendations`: List of recommendations

## Error Handling

The framework uses custom exceptions for better error handling:

```python
try:
    results = evaluator.evaluate(model)
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except EvaluationError as e:
    print(f"Evaluation failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Examples

See the [examples directory](../examples/) for complete working examples:

- [basic_evaluation.py](../examples/basic_evaluation.py) - Simple evaluation example
- [model_comparison.py](../examples/model_comparison.py) - Comparing multiple models
- [custom_scenarios.py](../examples/custom_scenarios.py) - Using custom scenarios
- [analysis_dashboard.ipynb](../visualizations/analysis_dashboard.ipynb) - Interactive analysis