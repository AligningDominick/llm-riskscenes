"""Unit tests for model implementations."""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock

from lmse.models.base import BaseModel, MockModel
from lmse.models.loader import ModelLoader
from lmse.models.claude import ClaudeModel
from lmse.models.openai import OpenAIModel


class TestBaseModel:
    """Test cases for BaseModel class."""
    
    def test_init(self):
        """Test base model initialization."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseModel("test", {})
    
    def test_mock_model_init(self):
        """Test mock model initialization."""
        model = MockModel("test-mock", {"responses": {"test": "response"}})
        assert model.name == "test-mock"
        assert model.responses == {"test": "response"}
    
    def test_mock_model_generate(self):
        """Test mock model generation."""
        responses = {
            "medical": "Medical response",
            "legal": "Legal response"
        }
        model = MockModel("mock", {"responses": responses, "default_response": "Default"})
        
        # Test predefined responses
        assert model.generate("medical") == "Medical response"
        assert model.generate("legal") == "Legal response"
        assert model.generate("unknown") == "Default"
        
        # Test automatic responses based on content
        assert "cannot provide medical advice" in model.generate("What medication for headache?").lower()
        assert "cannot provide legal advice" in model.generate("Can I sue my employer?").lower()
        assert "cannot provide financial advice" in model.generate("Should I invest in stocks?").lower()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        model = MockModel("mock", {"rate_limit": 2})  # 2 requests per minute
        
        # First two requests should work
        model._check_rate_limit()
        model._check_rate_limit()
        
        # Third request should be rate limited
        import time
        start = time.time()
        model._check_rate_limit()
        elapsed = time.time() - start
        
        # Should have waited
        assert elapsed < 1  # But not the full minute in test
    
    def test_retry_with_backoff(self):
        """Test retry logic."""
        model = MockModel("mock", {"retry_attempts": 3, "retry_delay": 0.1})
        
        # Function that fails twice then succeeds
        call_count = 0
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "Success"
        
        result = model._retry_with_backoff(flaky_func)
        assert result == "Success"
        assert call_count == 3
        
        # Function that always fails
        def always_fails():
            raise Exception("Permanent error")
        
        with pytest.raises(Exception, match="Permanent error"):
            model._retry_with_backoff(always_fails)


class TestModelLoader:
    """Test cases for ModelLoader class."""
    
    def test_load_mock_model(self):
        """Test loading mock model."""
        model = ModelLoader.load("mock", {"responses": {"test": "response"}})
        assert isinstance(model, MockModel)
        assert model.name == "mock"
    
    def test_load_unknown_model(self):
        """Test loading unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelLoader.load("unknown-model")
    
    def test_create_mock_model(self):
        """Test creating mock model helper."""
        model = ModelLoader.create_mock_model(
            name="test",
            responses={"q": "a"},
            default_response="default"
        )
        assert isinstance(model, MockModel)
        assert model.generate("q") == "a"
        assert model.generate("other") == "default"
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelLoader.list_available_models()
        assert isinstance(models, dict)
        assert "mock" in models
        assert "claude-3-opus" in models
        assert "gpt-4" in models
    
    def test_register_model(self):
        """Test registering custom model."""
        class CustomModel(BaseModel):
            def generate(self, prompt, **kwargs):
                return "Custom response"
            
            def validate_config(self):
                return True
        
        ModelLoader.register_model("custom", CustomModel)
        
        model = ModelLoader.load("custom", {})
        assert isinstance(model, CustomModel)
        assert model.generate("test") == "Custom response"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_load_claude_model(self):
        """Test loading Claude model."""
        with patch('anthropic.Anthropic'):
            model = ModelLoader.load("claude-3-opus", {"api_key": "test-key"})
            assert isinstance(model, ClaudeModel)
            assert model.name == "claude-3-opus"
            assert model.model_version == "claude-3-opus-20240229"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_load_openai_model(self):
        """Test loading OpenAI model."""
        with patch('openai.OpenAI'):
            model = ModelLoader.load("gpt-4", {"api_key": "test-key"})
            assert isinstance(model, OpenAIModel)
            assert model.name == "gpt-4"
            assert model.model_version == "gpt-4-turbo-preview"


class TestClaudeModel:
    """Test cases for ClaudeModel class."""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch('anthropic.Anthropic') as mock:
            yield mock
    
    @pytest.fixture
    def claude_model(self, mock_anthropic):
        """Create ClaudeModel instance."""
        return ClaudeModel("claude-3-opus", {"api_key": "test-key"})
    
    def test_init_without_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                ClaudeModel("claude-3-opus", {})
    
    def test_init_with_env_key(self, mock_anthropic):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            model = ClaudeModel("claude-3-opus", {})
            assert model.api_key == "env-key"
    
    def test_generate(self, claude_model, mock_anthropic):
        """Test response generation."""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated response")]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        response = claude_model.generate(
            prompt="Test prompt",
            system_prompt="Be helpful",
            temperature=0.5,
            max_tokens=100
        )
        
        assert response == "Generated response"
        
        # Verify API call
        mock_anthropic.return_value.messages.create.assert_called_once()
        call_args = mock_anthropic.return_value.messages.create.call_args
        assert call_args[1]['messages'] == [{"role": "user", "content": "Test prompt"}]
        assert call_args[1]['system'] == "Be helpful"
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_generate_async(self, claude_model):
        """Test async response generation."""
        with patch('anthropic.AsyncAnthropic') as mock_async:
            mock_response = Mock()
            mock_response.content = [Mock(text="Async response")]
            mock_async.return_value.messages.create = AsyncMock(return_value=mock_response)
            
            response = await claude_model.generate_async("Test prompt")
            assert response == "Async response"
    
    def test_get_info(self, claude_model):
        """Test model info."""
        info = claude_model.get_info()
        assert info['name'] == "claude-3-opus"
        assert info['provider'] == "Anthropic"
        assert info['supports_async'] is True
        assert info['max_context_length'] == 200000


class TestOpenAIModel:
    """Test cases for OpenAIModel class."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch('openai.OpenAI') as mock:
            yield mock
    
    @pytest.fixture
    def openai_model(self, mock_openai):
        """Create OpenAIModel instance."""
        return OpenAIModel("gpt-4", {"api_key": "test-key"})
    
    def test_generate(self, openai_model, mock_openai):
        """Test response generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated response"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        response = openai_model.generate(
            prompt="Test prompt",
            system_prompt="Be helpful",
            temperature=0.7
        )
        
        assert response == "Generated response"
        
        # Verify API call
        call_args = mock_openai.return_value.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Test prompt"}
    
    def test_get_info(self, openai_model):
        """Test model info."""
        info = openai_model.get_info()
        assert info['name'] == "gpt-4"
        assert info['provider'] == "OpenAI"
        assert info['supports_async'] is True
        assert info['max_context_length'] == 128000