"""OpenAI model implementation."""

import os
from typing import Dict, Optional, Any
import openai
from openai import AsyncOpenAI

from .base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI model implementation using OpenAI API."""
    
    MODEL_MAPPING = {
        "gpt-4": "gpt-4-turbo-preview",
        "gpt-3.5-turbo": "gpt-3.5-turbo-1106",
    }
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize OpenAI model.
        
        Args:
            name: Model identifier
            config: Model configuration
        """
        super().__init__(name, config)
        
        # Get API key
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set 'api_key' in config or OPENAI_API_KEY environment variable."
            )
        
        # Initialize client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
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
        """Generate a response using OpenAI.
        
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
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = self._retry_with_backoff(
                self.client.chat.completions.create,
                model=self.model_version,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract text from response
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using OpenAI asynchronously.
        
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
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Make async API call
            response = await self.async_client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract text from response
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def validate_config(self) -> bool:
        """Validate the OpenAI configuration.
        
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
            "provider": "OpenAI",
            "model_version": self.model_version,
            "supports_async": True,
            "max_context_length": 128000 if "gpt-4" in self.name else 16385,
        })
        return info