"""Unit tests for configuration module."""

import pytest
import json
import yaml
from pathlib import Path

from lmse.utils.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_init_empty(self):
        """Test initialization with empty data."""
        config = Config()
        assert config.data == {}
    
    def test_init_with_data(self):
        """Test initialization with data."""
        data = {"key": "value", "nested": {"key": "value"}}
        config = Config(data)
        assert config.data == data
    
    def test_get_simple(self):
        """Test getting simple values."""
        config = Config({"key": "value"})
        assert config.get("key") == "value"
        assert config.get("missing") is None
        assert config.get("missing", "default") == "default"
    
    def test_get_nested(self):
        """Test getting nested values with dot notation."""
        config = Config({
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        })
        assert config.get("level1.level2.level3") == "value"
        assert config.get("level1.level2") == {"level3": "value"}
        assert config.get("level1.missing.level3") is None
    
    def test_set_simple(self):
        """Test setting simple values."""
        config = Config()
        config.set("key", "value")
        assert config.get("key") == "value"
    
    def test_set_nested(self):
        """Test setting nested values."""
        config = Config()
        config.set("level1.level2.key", "value")
        assert config.get("level1.level2.key") == "value"
        assert config.data == {"level1": {"level2": {"key": "value"}}}
    
    def test_update(self):
        """Test updating configuration."""
        config = Config({"key1": "value1", "nested": {"key": "old"}})
        config.update({
            "key2": "value2",
            "nested": {"key": "new", "extra": "value"}
        })
        
        assert config.get("key1") == "value1"
        assert config.get("key2") == "value2"
        assert config.get("nested.key") == "new"
        assert config.get("nested.extra") == "value"
    
    def test_load_from_dict(self):
        """Test loading from dictionary."""
        data = {"key": "value"}
        config = Config.load(data)
        assert isinstance(config, Config)
        assert config.data == data
    
    def test_load_from_config(self):
        """Test loading from existing Config object."""
        original = Config({"key": "value"})
        config = Config.load(original)
        assert config is original
    
    def test_save_json(self, temp_dir):
        """Test saving to JSON file."""
        config = Config({"key": "value", "number": 42})
        json_path = temp_dir / "config.json"
        
        config.save(json_path)
        
        with open(json_path) as f:
            loaded = json.load(f)
        
        assert loaded == config.data
    
    def test_save_yaml(self, temp_dir):
        """Test saving to YAML file."""
        config = Config({"key": "value", "number": 42})
        yaml_path = temp_dir / "config.yaml"
        
        config.save(yaml_path)
        
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)
        
        assert loaded == config.data
    
    def test_load_json_file(self, temp_dir):
        """Test loading from JSON file."""
        data = {"key": "value", "nested": {"number": 42}}
        json_path = temp_dir / "config.json"
        
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        config = Config.load(json_path)
        assert config.data == data
    
    def test_load_yaml_file(self, temp_dir):
        """Test loading from YAML file."""
        data = {"key": "value", "nested": {"number": 42}}
        yaml_path = temp_dir / "config.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
        
        config = Config.load(yaml_path)
        assert config.data == data
    
    def test_load_invalid_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.load("non_existent.yaml")
    
    def test_load_invalid_format(self, temp_dir):
        """Test loading unsupported file format."""
        txt_path = temp_dir / "config.txt"
        txt_path.write_text("invalid")
        
        with pytest.raises(ValueError, match="Unsupported config format"):
            Config.load(txt_path)
    
    def test_env_variable_override(self, monkeypatch):
        """Test environment variable override."""
        config = Config({"model": {"temperature": 0.7}})
        
        # Set environment variable
        monkeypatch.setenv("LMSE_MODEL_TEMPERATURE", "0.5")
        
        # Environment variable should take precedence
        assert config.get("model.temperature") == 0.5
    
    def test_env_variable_parsing(self, monkeypatch):
        """Test environment variable type parsing."""
        config = Config()
        
        # Test boolean
        monkeypatch.setenv("LMSE_FEATURE_ENABLED", "true")
        assert config.get("feature.enabled") is True
        
        monkeypatch.setenv("LMSE_FEATURE_ENABLED", "false")
        assert config.get("feature.enabled") is False
        
        # Test integer
        monkeypatch.setenv("LMSE_BATCH_SIZE", "42")
        assert config.get("batch.size") == 42
        
        # Test float
        monkeypatch.setenv("LMSE_THRESHOLD", "0.95")
        assert config.get("threshold") == 0.95
        
        # Test JSON
        monkeypatch.setenv("LMSE_LANGUAGES", '["en", "es", "fr"]')
        assert config.get("languages") == ["en", "es", "fr"]
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config.default()
        
        # Check some expected defaults
        assert config.get("evaluation.async") is True
        assert config.get("model.temperature") == 0.7
        assert config.get("risk.thresholds.critical") == 80
        assert config.get("scoring.weights.safety") == 0.4
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = {"key": "value", "nested": {"key": "value"}}
        config = Config(data)
        result = config.to_dict()
        
        assert result == data
        # Ensure it's a copy
        result["new_key"] = "new_value"
        assert "new_key" not in config.data
    
    def test_repr(self):
        """Test string representation."""
        config = Config({"key": "value"})
        repr_str = repr(config)
        
        assert "Config" in repr_str
        assert "key" in repr_str
        assert "value" in repr_str