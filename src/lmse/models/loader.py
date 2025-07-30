"""Model loader for instantiating different LLM implementations."""

import importlib
from typing import Dict, Optional, Type, Union
from pathlib import Path

from .base import BaseModel, MockModel
from ..utils.config import Config


class ModelLoader:
    """Factory class for loading different model implementations."""
    
    # Registry of available models
    MODEL_REGISTRY = {
        "mock": MockModel,
        "claude-3-opus": "lmse.models.claude.ClaudeModel",
        "claude-3-sonnet": "lmse.models.claude.ClaudeModel",
        "claude-3-haiku": "lmse.models.claude.ClaudeModel",
        "gpt-4": "lmse.models.openai.OpenAIModel",
        "gpt-3.5-turbo": "lmse.models.openai.OpenAIModel",
        "gemini-pro": "lmse.models.google.GeminiModel",
        "gemini-ultra": "lmse.models.google.GeminiModel",
        "llama-2-70b": "lmse.models.huggingface.HuggingFaceModel",
        "mistral-large": "lmse.models.mistral.MistralModel",
    }
    
    @classmethod
    def load(
        cls,
        model_name: str,
        config: Optional[Union[Dict, Config, str, Path]] = None,
        **kwargs
    ) -> BaseModel:
        """Load a model by name.
        
        Args:
            model_name: Name of the model to load
            config: Model configuration (dict, Config object, or path)
            **kwargs: Additional configuration parameters
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If model name is not recognized
            ImportError: If model implementation cannot be imported
        """
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls.MODEL_REGISTRY.keys())}"
            )
        
        # Load configuration
        if config is None:
            config = {}
        elif isinstance(config, (str, Path)):
            config = Config.load(config)
        elif isinstance(config, Config):
            config = config.data
        
        # Merge with kwargs
        model_config = {**config, **kwargs}
        
        # Get model class
        model_spec = cls.MODEL_REGISTRY[model_name]
        
        if isinstance(model_spec, str):
            # Dynamic import
            model_class = cls._import_model_class(model_spec)
        else:
            # Direct class reference
            model_class = model_spec
        
        # Instantiate model
        return model_class(name=model_name, config=model_config)
    
    @classmethod
    def _import_model_class(cls, module_path: str) -> Type[BaseModel]:
        """Dynamically import a model class.
        
        Args:
            module_path: Full module path (e.g., 'lmse.models.claude.ClaudeModel')
            
        Returns:
            Model class
            
        Raises:
            ImportError: If module or class cannot be imported
        """
        try:
            module_name, class_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            
            # Validate it's a proper model class
            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"{module_path} is not a subclass of BaseModel"
                )
            
            return model_class
            
        except (ImportError, AttributeError, ValueError) as e:
            raise ImportError(
                f"Failed to import model class from {module_path}: {e}"
            )
    
    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Union[Type[BaseModel], str]
    ):
        """Register a new model implementation.
        
        Args:
            name: Model identifier
            model_class: Model class or import path
        """
        cls.MODEL_REGISTRY[name] = model_class
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List all available models.
        
        Returns:
            Dictionary mapping model names to their descriptions
        """
        models = {}
        for name, spec in cls.MODEL_REGISTRY.items():
            if isinstance(spec, str):
                models[name] = f"Dynamic import: {spec}"
            else:
                models[name] = f"Direct class: {spec.__name__}"
        
        return models
    
    @classmethod
    def create_mock_model(
        cls,
        name: str = "mock",
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mock response"
    ) -> MockModel:
        """Create a mock model for testing.
        
        Args:
            name: Model name
            responses: Dictionary of prompt->response mappings
            default_response: Default response for unmapped prompts
            
        Returns:
            Mock model instance
        """
        config = {
            "responses": responses or {},
            "default_response": default_response
        }
        return MockModel(name=name, config=config)