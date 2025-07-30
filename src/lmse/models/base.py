"""Base model interface for LLM implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import asyncio
import time


class BaseModel(ABC):
    """Abstract base class for all LLM model implementations."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the model.
        
        Args:
            name: Model identifier
            config: Model-specific configuration
        """
        self.name = name
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 1)
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 60)  # requests per minute
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Async version of generate method.
        
        Default implementation wraps the sync method.
        Override for native async support.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            prompt,
            system_prompt,
            temperature,
            max_tokens,
            kwargs
        )
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if rate limit exceeded
        if self.request_count >= self.rate_limit:
            sleep_time = self.rate_limit_window - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise
                
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the model configuration.
        
        Returns:
            True if configuration is valid
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and capabilities.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": {
                "timeout": self.timeout,
                "retry_attempts": self.retry_attempts,
                "rate_limit": self.rate_limit,
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class MockModel(BaseModel):
    """Mock model for testing purposes."""
    
    def __init__(self, name: str = "mock", config: Optional[Dict] = None):
        super().__init__(name, config or {})
        self.responses = config.get("responses", {}) if config else {}
        self.default_response = config.get("default_response", "Mock response") if config else "Mock response"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a mock response."""
        # Check for predefined responses
        if prompt in self.responses:
            return self.responses[prompt]
        
        # Generate based on prompt content
        if "medical" in prompt.lower() or "healthcare" in prompt.lower():
            return "I cannot provide medical advice. Please consult a healthcare professional."
        elif "legal" in prompt.lower():
            return "I cannot provide legal advice. Please consult a qualified attorney."
        elif "investment" in prompt.lower() or "financial" in prompt.lower():
            return "I cannot provide financial advice. Please consult a financial advisor."
        
        return self.default_response
    
    def validate_config(self) -> bool:
        """Mock model config is always valid."""
        return True