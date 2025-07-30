"""Claude model implementation."""

import os
from typing import Dict, Optional, Any
import anthropic
from anthropic import AsyncAnthropic

from .base import BaseModel


class ClaudeModel(BaseModel):
    """Claude model implementation using Anthropic API."""
    
    MODEL_MAPPING = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229", 
        "claude-3-haiku": "claude-3-haiku-20240307",
    }
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize Claude model.
        
        Args:
            name: Model identifier
            config: Model configuration
        """
        super().__init__(name, config)
        
        # Get API key
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set 'api_key' in config or ANTHROPIC_API_KEY environment variable."
            )
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        
        # Get model version
        self.model_version = self.MODEL_MAPPING.get(name, name)
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using Claude.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        self._check_rate_limit()
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            response = self._retry_with_backoff(
                self.client.messages.create,
                model=self.model_version,
                messages=messages,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using Claude asynchronously.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make async API call
            response = await self.async_client.messages.create(
                model=self.model_version,
                messages=messages,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")
    
    def validate_config(self) -> bool:
        """Validate the Claude configuration.
        
        Returns:
            True if configuration is valid
        """
        if not self.api_key:
            return False
        
        # Test API connection
        try:
            self.generate("Hello", max_tokens=10)
            return True
        except Exception:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_info()
        info.update({
            "provider": "Anthropic",
            "model_version": self.model_version,
            "supports_async": True,
            "max_context_length": 200000,  # Claude 3 context window
        })
        return info