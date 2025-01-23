from typing import TypeVar, Optional, Union, Type
import backoff
import json
from openai import OpenAI
from langfuse.openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ParsedChatCompletion
from pydantic import BaseModel
from .cache import CacheResult
from .exceptions import APICallError

class CachedChatCompletions:
    def __init__(self, 
                 client: Optional[OpenAI] = None, 
                 cache: Optional[CacheResult] = None):
        """
        Initialize the cached chat completions handler.
        
        :param client: OpenAI client (will create a default one if not provided)
        :param cache: AICache instance (will create a default one if not provided)
        """
        self.client = client or OpenAI()
        self.cache = cache or CacheResult()
        self.CACHE_MISS_SENTINEL = object()

    @backoff.on_exception(backoff.expo, Exception)
    async def _make_api_call(self, is_structured: bool, **kwargs):
        try:
            if is_structured:
                return await self.client.beta.chat.completions.parse(**kwargs)
            return await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise APICallError(f"API call failed: {str(e)}") from e

    async def get_completion(
            self,
            *,
            model: str,
            messages: list,
            response_format: Optional[Union[BaseModel, Type[BaseModel]]] = None,
            **kwargs
    ) -> Union[ChatCompletion, ParsedChatCompletion]:
        """
        Unified method for both structured and unstructured completions
        
        :param model: OpenAI model to use
        :param messages: List of message dictionaries
        :param response_format: Optional Pydantic model for structured output
        :param kwargs: Additional arguments to pass to the API
        :return: ChatCompletion or ParsedChatCompletion
        """
        is_structured = response_format is not None
        
        cache_key = self.cache.make_chat_completion_key(
            model=model,
            messages=messages,
            response_format=response_format if is_structured else None,
            **kwargs
        )

        cached_value = await self.cache.get_async(
            cache_key, 
            default=self.CACHE_MISS_SENTINEL
        )

        if cached_value is self.CACHE_MISS_SENTINEL:
            # Cache miss - make API call
            completion = await self._make_api_call(
                is_structured=is_structured,
                model=model,
                messages=messages,
                response_format=response_format if is_structured else None,
                **kwargs
            )
            await self.cache.set_async(cache_key, completion.json())
            return completion
        else:
            # Cache hit - parse response
            if is_structured:
                completion = ParsedChatCompletion.model_validate(
                    json.loads(cached_value)
                )
                # Handle structured response parsing
                for choice in completion.choices:
                    if not choice.message.refusal:
                        choice.message.parsed = response_format.model_validate(
                            choice.message.parsed
                        )
                return completion
            else:
                return ChatCompletion.model_validate(json.loads(cached_value))
            

    def extract_response(self, completion):
        """
        Unified method to extract response content for both structured and unstructured completions.
        
        :param completion: ChatCompletion or ParsedChatCompletion object
        :return: Extracted content (either string or parsed model)
        """
        # Unstructured completion
        if isinstance(completion, ChatCompletion):
            return completion.choices[0].message.content
        
        # Structured completion
        elif hasattr(completion, 'choices') and completion.choices:
            # For structured completions, return the parsed object
            return completion.choices[0].message.parsed
        
        # Fallback
        raise ValueError("Unsupported completion type")