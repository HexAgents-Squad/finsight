import json
import time
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, List, Dict, Any

from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIStatusError
from pydantic import BaseModel, ValidationError

from core.config import Settings
from core.exceptions import FinSightError

logger = logging.getLogger("finsight.llm")
T = TypeVar("T", bound=BaseModel)

class AbstractLLMClient(ABC):
    """Interface Segregation Principle: Abstract base contract for LLM operations."""
    
    @abstractmethod
    async def chat(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.1) -> str:
        pass

    @abstractmethod
    async def chat_json(self, messages: List[Dict[str, str]], response_model: Type[T], model: Optional[str] = None, max_retries: Optional[int] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass


class LLMClient(AbstractLLMClient):
    """Production Enterprise Async LLM Client using Dependency Injection and Dual Embeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES
        )
        self._st_model = None

    async def chat(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        model: Optional[str] = None, 
        temperature: float = 0.1
    ) -> str:
        target_model = model or self.settings.LLM_MODEL
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature
            )
            content = response.choices[0].message.content
            if not content:
                raise FinSightError(f"Empty payload returned from {target_model}", error_type="EmptyResponseError")
            return content.strip()
            
        except (RateLimitError, APIConnectionError) as e:
            logger.error(f"Network failure reaching LLM provider ({target_model}): {e}")
            raise FinSightError(f"Network error communicating with LLM: {str(e)}", error_type="NetworkError") from e
        except APIStatusError as e:
            raise FinSightError(f"LLM Provider status error: {e.message}", error_type=e.__class__.__name__) from e
        except Exception as e:
            raise FinSightError(f"Unexpected chat failure on model {target_model}: {str(e)}", error_type="ChatCompletionError") from e

    async def chat_json(
        self, 
        messages: List[Dict[str, str]], 
        response_model: Type[T], 
        model: Optional[str] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates structured JSON validated against a Pydantic model with robust regex extraction
        and schema validation self-correction.
        """
        target_model = model or self.settings.LLM_MODEL
        effective_retries = max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        
        # Enforce strict system-level instruction for local model compliance
        system_instruction = (
            "You are a precise data extraction engine. You must ALWAYS and ONLY return valid JSON "
            "matching the requested schema. Do not include conversational filler, greetings, or markdown explanations "
            "outside of standard JSON code fences."
        )

        schema_prompt = (
            f"\n\nYou must return output adhering strictly to this JSON Schema:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}\n"
            f"Output ONLY the JSON object inside ```json ... ``` code fences."
        )

        working_messages = []
        if not any(m.get("role") == "system" for m in messages):
            working_messages.append({"role": "system", "content": system_instruction})
        
        for m in messages:
            working_messages.append(dict(m))

        if working_messages and working_messages[-1]["role"] == "user":
            working_messages[-1]["content"] += schema_prompt
        else:
            working_messages.append({"role": "user", "content": schema_prompt})

        start_time = time.perf_counter()
        total_tokens = 0
        content = ""

        for attempt in range(effective_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=target_model,
                    messages=working_messages,
                    temperature=0.0
                )

                if response.usage:
                    total_tokens += response.usage.total_tokens

                raw_content = response.choices[0].message.content or ""
                content = raw_content.strip()

                # Robust Extraction: Try stripping code fences first, then fallback to regex finding `{...}`
                clean_content = content
                if "```json" in clean_content:
                    parts = clean_content.split("```json")
                    if len(parts) > 1:
                        clean_content = parts[1].split("```")[0]
                elif "```" in clean_content:
                    parts = clean_content.split("```")
                    if len(parts) > 1:
                        clean_content = parts[1]
                
                clean_content = clean_content.strip()

                # Regex fallback if extra conversational chatter still exists around JSON
                if not (clean_content.startswith("{") and clean_content.endswith("}")):
                    json_match = re.search(r"\{.*\}", clean_content, re.DOTALL)
                    if json_match:
                        clean_content = json_match.group(0)

                parsed = json.loads(clean_content)
                validated = response_model.model_validate(parsed)
                
                latency = round(time.perf_counter() - start_time, 4)

                logger.info(
                    "JSON Generation Success",
                    extra={
                        "model": target_model,
                        "latency_s": latency,
                        "tokens": total_tokens,
                        "retries": attempt
                    }
                )

                return {
                    "data": validated,
                    "metadata": {
                        "retries_used": attempt,
                        "latency_seconds": latency,
                        "total_tokens": total_tokens
                    }
                }

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Schema validation attempt {attempt+1}/{effective_retries+1} failed for {target_model}: {e}")
                
                if attempt >= effective_retries:
                    raise FinSightError(
                        f"Failed JSON structural contract after {effective_retries} validation retries: {e}",
                        error_type="JSONValidationError"
                    ) from e

                # Self-correction prompt injection with explicit reminder
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({
                    "role": "user",
                    "content": f"Your previous output failed validation with error: {str(e)}. "
                               f"Please output ONLY raw valid JSON matching the schema, with no extra text."
                })
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                logger.warning(f"API/Network error during chat_json attempt {attempt+1}/{effective_retries+1}: {e}")
                
                if attempt >= effective_retries:
                    raise FinSightError(
                        f"API error during chat_json generation after {effective_retries} retries: {str(e)}",
                        error_type="ChatJsonAPIError"
                    ) from e
                
                await asyncio.sleep(min(2 ** attempt, 6))
            except Exception as e:
                if attempt >= effective_retries:
                    raise FinSightError(
                        f"Unexpected error during chat_json generation: {str(e)}",
                        error_type="UncaughtStructuredError"
                    ) from e

        raise FinSightError("Maximum retries exhausted for chat_json.", error_type="RetryExceededError")

    async def embed(self, text: str) -> List[float]:
        """
        Generates text embeddings using either a local sentence-transformers model 
        (executed on a background thread to prevent async loop blocking) or an OpenAI-compatible endpoint.
        """
        backend = self.settings.EMBED_BACKEND.strip().lower()

        try:
            if backend == "sentence-transformers":
                if self._st_model is None:
                    from sentence_transformers import SentenceTransformer
                    # Lazy load model in a thread executor to prevent blocking event loop on import/init
                    loop = asyncio.get_running_loop()
                    self._st_model = await loop.run_in_executor(
                        None, 
                        SentenceTransformer, 
                        self.settings.EMBED_LOCAL_MODEL_NAME
                    )
                
                # Run CPU-bound encoding inside an executor thread
                loop = asyncio.get_running_loop()
                vector = await loop.run_in_executor(
                    None, 
                    self._st_model.encode, 
                    text
                )
                return vector.tolist()
            
            else:
                # Default to OpenAI-compatible endpoint
                response = await self.client.embeddings.create(
                    model=self.settings.EMBED_MODEL,
                    input=text
                )
                return response.data[0].embedding

        except Exception as e:
            raise FinSightError(
                f"Embedding generation failed via backend '{backend}': {str(e)}", 
                error_type="EmbeddingError"
            ) from e