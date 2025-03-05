import requests
import json
import asyncio
from typing import Optional, List, AsyncGenerator
from utils.config import LLM_MODE, LLM_ENDPOINT, TOKEN, MODEL_NAME_API, MODEL_NAME_LOCAL, DEVICE, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES, LOG_LEVEL
from utils.logger import AsyncLogger
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch

# Set log level from config
logger.remove()
logger.add(lambda msg: print(msg), level=LOG_LEVEL, format="{time} {level} {message}")

class LLMClient:
    """
    A client for interacting with LLMs, supporting both Phoenix API endpoints and local models,
    with multimodal capabilities (text and image processing).
    """
    def __init__(self, endpoint_type: str = "default", workspace_a: str = "nb", workspace_b: str = "km74h"):
        """
        Initialize the LLMClient with configurable endpoint type and workspace parameters.
        
        Args:
            endpoint_type (str): Type of endpoint to use ("default" for standard Phoenix, "phoenix_alt" for the alternate Phoenix endpoint from the image).
            workspace_a (str): Workspace parameter 'a' for the alternate Phoenix endpoint (default: "nb").
            workspace_b (str): Workspace parameter 'b' for the alternate Phoenix endpoint (default: "km74h").
        """
        self.mode = LLM_MODE
        self.endpoint_type = endpoint_type.lower()
        
        # Configure endpoints
        if self.endpoint_type == "default":
            self.base_url = LLM_ENDPOINT  # Standard Phoenix endpoint (e.g., /v1/chat/completions)
            self.model = MODEL_NAME_API
        elif self.endpoint_type == "phoenix_alt":
            self.base_url = "http://1rche002papd.sdi.corp.bankofamerica.com:8123/v1"  # Alternate Phoenix endpoint from image
            self.model = f"/phoenix/workspaces/{workspace_a}/{workspace_b}/llama3.3-4bit-awq"
        else:
            raise ValueError(f"Unknown endpoint_type: {endpoint_type}. Use 'default' or 'phoenix_alt'.")
        
        self.headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Local model setup
        if self.mode == "local":
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LOCAL)
            self.model_local = AutoModelForCausalLM.from_pretrained(MODEL_NAME_LOCAL).to(DEVICE)
            self.text_pipeline = pipeline(
                "text-generation",
                model=self.model_local,
                tokenizer=self.tokenizer,
                device=DEVICE
            )
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def generate(self, prompt: str, image_base64: Optional[str] = None, max_length: int = MAX_LENGTH, temperature: float = TEMPERATURE, top_p: float = TOP_P, stop: List[str] = STOP_SEQUENCES) -> str:
        """
        Generate a response using the selected LLM endpoint or local model, optionally with image context.
        
        Args:
            prompt (str): The input prompt or query.
            image_base64 (Optional[str]): Base64-encoded image for multimodal input (if applicable).
            max_length (int): Maximum length of the response (tokens).
            temperature (float): Sampling temperature for response creativity.
            top_p (float): Nucleus sampling parameter.
            stop (List[str]): Stop sequences for generation.
        
        Returns:
            str: Generated response from the LLM.
        """
        start_time = time.time()
        try:
            if self.mode == "api":
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop
                }
                if image_base64 and self.endpoint_type == "default":
                    # Add image context for default Phoenix endpoint (assuming it supports multimodal input)
                    payload["image"] = image_base64
                
                response = await asyncio.to_thread(
                    requests.post,
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                response.raise_for_status()
                response_data = response.json()
                answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response received")
            else:  # Local mode
                if image_base64:
                    # Process image for multimodal input
                    image = await asyncio.to_thread(self.image_processor.decode_base64_to_pil, image_base64)
                    image_inputs = self.image_processor(images=image, return_tensors="pt").to(DEVICE)
                    image_features = self.image_model.get_image_features(**image_inputs).detach().cpu().numpy()[0]
                    text_features = self.text_embedder.encode(prompt)
                    combined_features = np.concatenate([text_features, image_features])
                    prompt = f"Prompt with image context: {prompt} [Image features included]"
                
                response = self.text_pipeline(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequence=stop
                )
                answer = response[0]["generated_text"]
            
            duration = time.time() - start_time
            await AsyncLogger.info(f"Generated response in {duration:.2f} seconds for prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            return answer
        except Exception as e:
            await AsyncLogger.error(f"Error generating response for prompt {prompt}: {e}")
            return f"Error: {str(e)}"

    async def generate_stream(self, prompt: str, image_base64: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a response token by token for real-time display, optionally with image context.
        
        Args:
            prompt (str): The input prompt or query.
            image_base64 (Optional[str]): Base64-encoded image for multimodal input (if applicable).
        
        Yields:
            str: Tokens of the response as they are generated.
        """
        start_time = time.time()
        try:
            if self.mode == "api":
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_LENGTH,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "stream": True
                }
                if image_base64 and self.endpoint_type == "default":
                    payload["image"] = image_base64
                
                response = await asyncio.to_thread(
                    requests.post,
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30,
                    stream=True
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8").replace("data: ", "")
                        if decoded_line == "[DONE]":
                            break
                        data = json.loads(decoded_line)
                        token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            yield token
            else:  # Local mode
                if image_base64:
                    image = await asyncio.to_thread(self.image_processor.decode_base64_to_pil, image_base64)
                    image_inputs = self.image_processor(images=image, return_tensors="pt").to(DEVICE)
                    image_features = self.image_model.get_image_features(**image_inputs).detach().cpu().numpy()[0]
                    text_features = self.text_embedder.encode(prompt)
                    combined_features = np.concatenate([text_features, image_features])
                    prompt = f"Prompt with image context: {prompt} [Image features included]"
                
                for token in self.text_pipeline(
                    prompt,
                    max_length=MAX_LENGTH,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    return_full_text=False,
                    return_dict_in_generate=True
                )[0]["token_ids"]:
                    yield self.tokenizer.decode(token, skip_special_tokens=True)
                    await asyncio.sleep(0.01)  # Simulate streaming delay
            
            duration = time.time() - start_time
            await AsyncLogger.info(f"Streamed response in {duration:.2f} seconds for prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        except Exception as e:
            await AsyncLogger.error(f"Error streaming response for prompt {prompt}: {e}")
            yield f"Error: {str(e)}"

    async def classify(self, text: str, labels: List[str]) -> dict:
        """
        Classify text into one of the provided labels using the LLM.
        
        Args:
            text (str): Input text to classify.
            labels (List[str]): List of possible labels.
        
        Returns:
            dict: Scores for each label (higher score indicates higher likelihood).
        """
        start_time = time.time()
        prompt = f"Classify the following text into one of these labels: {', '.join(labels)}. Text: {text}\nReturn a JSON object with scores for each label (0-1)."
        try:
            response = await self.generate(prompt, max_length=100)
            scores = json.loads(response)
            duration = time.time() - start_time
            await AsyncLogger.info(f"Classified text in {duration:.2f} seconds: {text[:50]}{'...' if len(text) > 50 else ''}")
            return scores
        except Exception as e:
            await AsyncLogger.error(f"Error classifying text {text}: {e}")
            return {label: 0.0 for label in labels}

    async def process_image(self, image_base64: str) -> str:
        """
        Process an image and return a textual description using CLIP and the LLM.
        
        Args:
            image_base64 (str): Base64-encoded image string.
        
        Returns:
            str: Textual description of the image.
        """
        start_time = time.time()
        try:
            image = await asyncio.to_thread(self.image_processor.decode_base64_to_pil, image_base64)
            inputs = self.image_processor(images=image, return_tensors="pt").to(DEVICE)
            image_features = self.image_model.get_image_features(**inputs).detach().cpu().numpy()[0]
            
            prompt = "Describe this image: [Image features included]"
            if self.mode == "api" and self.endpoint_type == "default":
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P
                }
                response = await asyncio.to_thread(
                    requests.post,
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                response.raise_for_status()
                response_data = response.json()
                description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No description available")
            else:  # Local mode
                description = self.text_pipeline(
                    prompt,
                    max_length=100,
                    temperature=TEMPERATURE,
                    top_p=TOP_P
                )[0]["generated_text"]
            
            duration = time.time() - start_time
            await AsyncLogger.info(f"Processed image description in {duration:.2f} seconds")
            return description
        except Exception as e:
            await AsyncLogger.error(f"Error processing image: {e}")
            return "Error processing image"
