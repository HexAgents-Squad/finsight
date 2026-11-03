import sys
import asyncio
import logging
from pathlib import Path

# Add project root to python execution path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from core.config import get_settings
from core.llm import LLMClient
from core.exceptions import FinSightError

# Configure standard structured execution logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class FilingExtractTest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    year: int = Field(..., description="Reporting fiscal year")
    summary: str = Field(..., description="One-sentence financial summary")

async def main() -> None:
    print("🚀 Initializing FinSight WP7 Enterprise Smoke Test...")
    
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded successfully:")
        print(f"   • LLM Endpoint: {settings.LLM_BASE_URL} ({settings.LLM_MODEL})")
        print(f"   • Embedding Backend: {settings.EMBED_BACKEND}")
        if settings.EMBED_BACKEND.lower() == "sentence-transformers":
            print(f"   • Local Embedding Model: {settings.EMBED_LOCAL_MODEL_NAME}")
        else:
            print(f"   • API Embedding Model: {settings.EMBED_MODEL}")
        print(f"   • SEC User-Agent: {settings.SEC_USER_AGENT}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

    # Dependency Injection: Injecting settings into LLMClient
    llm = LLMClient(settings=settings)

    try:
        # 1. Test Async Chat
        print("\n[1/3] Testing async chat completion...")
        reply = await llm.chat(prompt="Respond with 'OPERATIONAL' if you are online.")
        print(f"✅ Chat Response: {reply}")

        # 2. Test Dual-Backend Embeddings
        print(f"\n[2/3] Testing embedding generation (Backend: {settings.EMBED_BACKEND})...")
        vector = await llm.embed(text="Apple Inc. Q4 2023 10-K filing net sales analysis.")
        print(f"✅ Embedding generated successfully. Dimensions: {len(vector)}")

        # 3. Test Structured chat_json with Metrics & Self-Correction
        print("\n[3/3] Testing chat_json with validation, retries & metrics...")
        messages = [{"role": "user", "content": "Extract details: Microsoft Corporation filed its FY2023 annual report."}]
        result = await llm.chat_json(messages=messages, response_model=FilingExtractTest)
        
        print(f"✅ Structured Output Validated:")
        print(result["data"].model_dump_json(indent=2))
        print(f"📊 Execution Metrics: {result['metadata']}")

        print("\n🎉 All WP7 enterprise smoke tests passed successfully!")

    except FinSightError as fe:
        print(f"\n❌ FinSight Error [{fe.error_type}]: {fe.message}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())