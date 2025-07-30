"""Configuration management for the evaluation framework."""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Config:
    """Configuration manager for the evaluation framework."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            data: Configuration dictionary
        """
        self.data = data or {}
        self._env_prefix = "LMSE_"
    
    @classmethod
    def load(cls, source: Union[str, Path, Dict]) -> "Config":
        """Load configuration from various sources.
        
        Args:
            source: Configuration source (file path, dict, or Config object)
            
        Returns:
            Config object
        """
        if isinstance(source, cls):
            return source
        
        if isinstance(source, dict):
            return cls(source)
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            
            if path.suffix == ".json":
                with open(path, 'r') as f:
                    data = json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
            return cls(data)
        
        raise TypeError(f"Invalid configuration source type: {type(source)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check environment variable first
        env_key = self._env_prefix + key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Navigate nested dictionary
        keys = key.split(".")
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        target = self.data
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set value
        target[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self.data, updates)
    
    def _deep_update(self, target: Dict, updates: Dict) -> None:
        """Deep update nested dictionaries.
        
        Args:
            target: Target dictionary
            updates: Updates to apply
        """
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value.
        
        Args:
            value: String value from environment
            
        Returns:
            Parsed value
        """
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Check for boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False
        
        # Check for number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path, 'w') as f:
                json.dump(self.data, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.data.copy()
    
    def __repr__(self) -> str:
        return f"Config({json.dumps(self.data, indent=2)})"
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration.
        
        Returns:
            Default configuration
        """
        return cls({
            "evaluation": {
                "async": True,
                "batch_size": 10,
                "timeout": 60,
                "retry_attempts": 3,
            },
            "model": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.95,
            },
            "risk": {
                "thresholds": {
                    "critical": 80,
                    "high": 60,
                    "medium": 40,
                },
            },
            "scoring": {
                "weights": {
                    "safety": 0.4,
                    "compliance": 0.3,
                    "cultural": 0.3,
                },
            },
            "output": {
                "directory": "results",
                "format": "json",
                "include_responses": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "data": {
                "scenarios_dir": "datasets/scenarios",
                "cache_enabled": True,
            },
        })