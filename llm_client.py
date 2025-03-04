import requests
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from typing import Optional, List, Dict, Union, AsyncGenerator
from document_ai_agents.logger import logger
from utils.config import LLM_MODE, LLM_ENDPOINT, TOKEN, MODEL_NAME_API, MODEL_NAME_LOCAL, DEVICE, MAX_LENGTH, TEMPERATURE, TOP_P, STOP_SEQUENCES
from utils.image_utils import base64_to_pil_image
import asyncio
import json

class LLMClient:
    def __init__(self):
        self.mode = LLM_MODE
        if self.mode == "api":
            self.headers = {
                "User-Agent": "agentic_rag_framework",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}"
            }
            # Assume Phoenix supports multimodal prompts (adjust if not)
        else:  # local mode
            # Text + Image models
            self.text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_LOCAL)
            self.text_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_LOCAL).to(DEVICE)
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            self.text_generator = pipeline("text-generation", model=self.text_model, tokenizer=self.text_tokenizer, device=DEVICE if DEVICE == "cuda" else -1)

    def _format_messages(self, prompt: Union[str, Dict], role: str = "user", image_base64: Optional[str] = None) -> List[Dict]:
        """Format the prompt and optional image for API or local mode."""
        if self.mode == "api":
            messages = [{"role": role, "content": prompt}]
            if image_base64:
                messages[0]["content"] = {"text": prompt, "image": image_base64}  # Assume Phoenix supports multimodal JSON
            return messages
        else:
            return [{"role": role, "content": prompt}]

    def process_image(self, image_base64: str) -> str:
        """Process an image for multimodal input (local mode)."""
        if self.mode == "api":
            # Assume Phoenix handles images (adjust if not)
            return image_base64
        image = base64_to_pil_image(image_base64)
        inputs = self.vision_processor(images=image, return_tensors="pt").to(DEVICE)
        out = self.vision_model.generate(**inputs, max_length=50)
        return self.vision_processor.decode(out[0], skip_special_tokens=True)

    def generate(self, prompt: Union[str, Dict], image_base64: Optional[str] = None, max_length: int = MAX_LENGTH, temperature: float = TEMPERATURE, top_p: float = TOP_P, stop: List[str] = STOP_SEQUENCES) -> Optional[str]:
        """Generate text using either API or local model with multimodal support."""
        if self.mode == "api":
            try:
                messages = self._format_messages(prompt, image_base64=image_base64)
                request_body = {
                    "model": MODEL_NAME_API,
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop": stop,
                    "messages": messages
                }
                response = requests.post(LLM_ENDPOINT, json=request_body, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Failed to call Phoenix Generative AI Service: {e}")
                return None
        else:  # local mode
            try:
                if image_base64:
                    image_caption = self.process_image(image_base64)
                    prompt = f"Image description: {image_caption}\n{prompt}"
                response = self.text_generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop
                )[0]["generated_text"]
                return response[len(prompt):].strip()  # Remove prompt from response
            except Exception as e:
                logger.error(f"Failed to generate with local model: {e}")
                return None

    async def generate_stream(self, prompt: Union[str, Dict], image_base64: Optional[str] = None, max_length: int = MAX_LENGTH, temperature: float = TEMPERATURE, top_p: float = TOP_P, stop: List[str] = STOP_SEQUENCES) -> AsyncGenerator[str, None]:
        """Asynchronously stream text token by token using either API or local model with multimodal support."""
        if self.mode == "api":
            # Simulate streaming if Phoenix supports it (check API docs for actual streaming)
            response = await asyncio.to_thread(self.generate, prompt, image_base64, max_length, temperature, top_p, stop)
            for token in response.split():
                yield token + " "
                await asyncio.sleep(0.05)
        else:  # local mode
            try:
                if image_base64:
                    image_caption = await asyncio.to_thread(self.process_image, image_base64)
                    prompt = f"Image description: {image_caption}\n{prompt}"
                for output in self.text_generator(prompt, return_full_text=False, stream=True):
                    yield output["token_str"] + " "
                    await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Failed to stream with local model: {e}")
                yield "Error: Streaming failed"

    def classify(self, text: str, labels: list[str], image_base64: Optional[str] = None) -> dict[str, float]:
        """Classify text (and optionally an image) into labels using the LLM."""
        prompt = (
            f"Classify the following {'text and image' if image_base64 else 'text'} into one of these labels: {', '.join(labels)}. "
            f"Text: {text}\n{'Image: [provided]' if image_base64 else ''}\n"
            "Return a JSON object with probabilities for each label, e.g., {'label1': 0.8, 'label2': 0.1, 'label3': 0.1}."
        )
        response = self.generate(prompt, image_base64, max_length=200)
        if response:
            try:
                return json.loads(response)
            except Exception as e:
                logger.error(f"Failed to parse classification response: {e}")
        return {label: 1.0 / len(labels) for label in labels}  # Fallback uniform distribution

    async def plan_task(self, query: str) -> List[Dict]:
        """Generate a hierarchical task plan for agentic reasoning."""
        prompt = (
            f"Break down this query into sub-tasks for an agentic RAG framework: {query}\n"
            "Return a JSON list of tasks, each with 'agent' (agent name), 'subtask' (description), and 'priority' (0-10). "
            "Agents can be: document_qa_agent, term_sheet_agent, db_assistant_agent, rag_agent, document_multi_tool_agent."
        )
        response = await asyncio.to_thread(self.generate, prompt, max_length=MAX_LENGTH, temperature=0.1, top_p=0.9, stop=STOP_SEQUENCES)
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse task plan: {e}")
            return [{"agent": "rag_agent", "subtask": query, "priority": 5}]

    async def reflect(self, query: str, response: str, feedback: Optional[str] = None) -> dict:
        """Reflect on an interaction to improve agent performance."""
        prompt = (
            f"Query: {query}\nResponse: {response}\nFeedback: {feedback or 'None'}\n"
            "Reflect on this interaction: Identify errors, suggest improvements, and propose a refined strategy. "
            "Return a JSON object with 'errors', 'improvements', and 'strategy'."
        )
        response = await asyncio.to_thread(self.generate, prompt, max_length=MAX_LENGTH, temperature=0.5, top_p=0.9, stop=STOP_SEQUENCES)
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse reflection: {e}")
            return {"errors": "None", "improvements": "None", "strategy": "Maintain current approach"}
