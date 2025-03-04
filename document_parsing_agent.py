from langgraph.graph import END, START, StateGraph, Send
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import operator
import asyncio
from utils.llm_client import LLMClient
from document_utils import extract_images_from_pdf, extract_text_from_pdf
from image_utils import pil_image_to_base64_jpeg, base64_to_pil_image
from document_ai_agents.logger import logger
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.config import DEVICE

class DetectedLayoutItem(BaseModel):
    element_type: Literal["Table", "Figure", "Image", "Text-block"] = Field(
        ..., description="Type of detected Item. Find Tables, figures, and images. Use Text-Block for everything else, be as exhaustive as possible. Return 10 Items at most."
    )
    summary: str = Field(..., description="A detailed description of the layout Item.")
    coordinates: Optional[dict] = Field(None, description="Bounding box coordinates if applicable")

class LayoutElements(BaseModel):
    layout_items: list[DetectedLayoutItem] = Field(default_factory=list)

class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_base64_jpeg_images: list[str] = Field(default_factory=list)
    pages_as_text: list[str] = Field(default_factory=list)
    documents: Annotated[list[Document], operator.add] = Field(default_factory=list)

class FindLayoutItemsInput(BaseModel):
    document_path: str
    base64_jpeg: str
    page_text: str
    page_number: int

@agent_registry.register("parsing_agent")
class DocumentParsingAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
        self.graph = self.build_agent()

    async def get_images_and_text(self, state: DocumentLayoutParsingState):
        assert Path(state.document_path).is_file(), "File does not exist"
        # Async extraction
        images = await asyncio.to_thread(extract_images_from_pdf, state.document_path)
        texts = await asyncio.to_thread(extract_text_from_pdf, state.document_path)
        assert images, "No images extracted"
        state.pages_as_base64_jpeg_images = [pil_image_to_base64_jpeg(x) for x in images]
        state.pages_as_text = texts
        return state

    async def continue_to_find_layout_items(self, state: DocumentLayoutParsingState):
        return [
            Send(
                "find_layout_items",
                FindLayoutItemsInput(
                    base64_jpeg=base64_jpeg,
                    page_text=text,
                    page_number=i,
                    document_path=state.document_path,
                ),
            )
            for i, (base64_jpeg, text) in enumerate(zip(state.pages_as_base64_jpeg_images, state.pages_as_text))
        ]

    async def find_layout_items(self, state: FindLayoutItemsInput):
        logger.info(f"Processing page {state.page_number + 1}")
        image = base64_to_pil_image(state.base64_jpeg)
        # Multimodal analysis with text and image
        image_caption = await asyncio.to_thread(self.process_image, image)
        prompt = (
            f"Analyze this page: Text: {state.page_text}\nImage Description: {image_caption}\n"
            "Identify layout items (Tables, Figures, Images, Text-blocks) with summaries and coordinates if applicable. "
            "Return JSON: {LayoutElements.model_json_schema()}"
        )
        response = await asyncio.to_thread(self.llm_client.generate, prompt, image_base64=state.base64_jpeg)
        try:
            import json
            layout_items = LayoutElements(**json.loads(response))
            documents = [
                Document(
                    page_content=item.summary,
                    metadata={
                        "page_number": state.page_number,
                        "element_type": item.element_type,
                        "coordinates": item.coordinates,
                        "document_path": state.document_path,
                        "image_base64": state.base64_jpeg if item.element_type == "Image" else None,
                    },
                )
                for item in layout_items.layout_items
            ]
            return {"documents": documents}
        except Exception as e:
            logger.error(f"Failed to parse layout items: {e}")
            return {"documents": [Document(page_content="Default text-block", metadata={"page_number": state.page_number, "element_type": "Text-block", "document_path": state.document_path})]}

    def process_image(self, image):
        inputs = self.vision_processor(images=image, return_tensors="pt").to(DEVICE)
        out = self.vision_model.generate(**inputs)
        return self.vision_processor.decode(out[0], skip_special_tokens=True)

    def build_agent(self):
        builder = StateGraph(DocumentLayoutParsingState)
        builder.add_node("get_images_and_text", self.get_images_and_text)
        builder.add_node("find_layout_items", self.find_layout_items)
        builder.add_edge(START, "get_images_and_text")
        builder.add_conditional_edges("get_images_and_text", self.continue_to_find_layout_items)
        builder.add_edge("find_layout_items", END)
        return builder.compile()
