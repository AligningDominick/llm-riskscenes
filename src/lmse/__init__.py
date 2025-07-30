"""LLM Multilingual Safety Evaluation Framework.

A comprehensive framework for evaluating Large Language Model safety
across multiple languages and cultural contexts.
"""

__version__ = "1.0.0"
__author__ = "Safety Evaluation Team"
__email__ = "safety-eval@example.org"

from .core.evaluator import SafetyEvaluator
from .models.loader import ModelLoader
from .analyzers.risk_analyzer import RiskAnalyzer
from .utils.config import Config

__all__ = [
    "SafetyEvaluator",
    "ModelLoader",
    "RiskAnalyzer",
    "Config",
]

# Package metadata
SUPPORTED_MODELS = [
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "gpt-4",
    "gpt-3.5-turbo",
    "gemini-pro",
    "gemini-ultra",
    "llama-2-70b",
    "mistral-large",
]

SUPPORTED_LANGUAGES = [
    # High-resource languages
    "english", "chinese", "spanish", "french", "german", "japanese",
    # Medium-resource languages
    "hindi", "arabic", "portuguese", "russian", "korean",
    # Low-resource languages
    "swahili", "yoruba", "bengali", "vietnamese", "thai",
    "urdu", "turkish", "polish", "dutch", "italian",
]

EVALUATION_DOMAINS = [
    "healthcare",
    "education",
    "legal",
    "finance",
    "civic",
    "social",
    "technology",
    "environment",
]