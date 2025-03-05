import requests
import json
import asyncio
from utils.config import LLM_ENDPOINT, TOKEN, LOG_LEVEL, MODEL_NAME_API
from utils.logger import AsyncLogger
from loguru import logger

# Set log level from config
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

async def test_phoenix_connection():
    """
    Test the connection to the Phoenix Generative AI Service endpoint and load a model.
    """
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Sample query to test the model
    payload = {
        "model": MODEL_NAME_API,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        # Send POST request to Phoenix endpoint
        response = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the response
        response_data = response.json()
        answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response received")

        # Log success
        await AsyncLogger.info(f"Successfully connected to Phoenix endpoint at {LLM_ENDPOINT}")
        await AsyncLogger.info(f"Model response: {answer}")
        
        return {
            "status": "success",
            "endpoint": LLM_ENDPOINT,
            "model": MODEL_NAME_API,
            "response": answer
        }
    
    except requests.exceptions.RequestException as e:
        await AsyncLogger.error(f"Failed to connect to Phoenix endpoint: {e}")
        return {
            "status": "error",
            "endpoint": LLM_ENDPOINT,
            "model": MODEL_NAME_API,
            "error": str(e)
        }
    except json.JSONDecodeError as e:
        await AsyncLogger.error(f"Failed to parse Phoenix response: {e}")
        return {
            "status": "error",
            "endpoint": LLM_ENDPOINT,
            "model": MODEL_NAME_API,
            "error": str(e)
        }

def run_test():
    """
    Run the async test synchronously for convenience.
    """
    result = asyncio.run(test_phoenix_connection())
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_test()
