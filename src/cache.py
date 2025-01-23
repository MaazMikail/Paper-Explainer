import asyncio
import json
import os
from hashlib import md5
from diskcache import Cache
from pydantic import BaseModel
from typing import Optional

class CacheResult:
    def __init__(self, directory=".cached_data"):
        os.makedirs(directory, exist_ok=True)
        self.cache = Cache(directory=directory)

    async def set_async(self, key, val, **kwargs):
        return await asyncio.to_thread(self.cache.set, key, val, **kwargs)

    async def get_async(self, key, default=None, **kwargs):
        return await asyncio.to_thread(self.cache.get, key, default, **kwargs)

    @staticmethod
    def make_cache_key(key_name, **kwargs):
        kwargs_string = json.dumps(kwargs, sort_keys=True)
        kwargs_hash = md5(kwargs_string.encode('utf-8')).hexdigest()
        return f"{key_name}__{kwargs_hash}"

    def make_chat_completion_key(
            self,
            *,
            model: str,
            messages: list,
            response_format: Optional[BaseModel] = None,
            **kwargs
    ) -> str:
        base_params = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        if response_format:
            base_params["response_format"] = response_format.model_json_schema()
            key_prefix = "openai_chat_completion_structured"
        else:
            key_prefix = "openai_chat_completion"
            
        return self.make_cache_key(key_prefix, **base_params)