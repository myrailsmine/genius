from pydantic import BaseModel, Json
from typing import Any, Dict, Union
from document_ai_agents.logger import logger
import json

def replace_value_in_dict(item: Any, original_schema: Dict) -> Any:
    """
    Recursively replace $ref references in a schema with their resolved values from the original schema.
    
    Args:
        item (Any): The item (dict, list, or primitive) to process.
        original_schema (Dict): The original schema containing definitions for $ref references.
    
    Returns:
        Any: The processed item with $ref references resolved.
    
    Raises:
        KeyError: If a $ref reference cannot be resolved in the schema.
        ValueError: If the schema structure is invalid.
    """
    if isinstance(item, list):
        return [replace_value_in_dict(i, original_schema) for i in item]
    elif isinstance(item, dict):
        if list(item.keys()) == ["$ref"]:
            ref_path = item["$ref"][2:].split("/")  # Remove '#/definitions/' prefix
            current = original_schema.copy()
            for segment in ref_path:
                if not isinstance(current, dict) or segment not in current:
                    raise KeyError(f"Cannot resolve $ref path: {item['$ref']}")
                current = current[segment]
            return current
        else:
            return {
                key: replace_value_in_dict(value, original_schema)
                for key, value in item.items()
            }
    return item

def delete_keys_recursive(data: Any, keys_to_delete: List[str]) -> Any:
    """
    Recursively delete specified keys from a dictionary or list structure.
    
    Args:
        data (Any): The data structure (dict, list, or primitive) to process.
        keys_to_delete (List[str]): List of keys to remove from dictionaries.
    
    Returns:
        Any: The modified data structure with specified keys removed.
    """
    if isinstance(data, dict):
        return {
            key: delete_keys_recursive(value, keys_to_delete)
            for key, value in data.items()
            if key not in keys_to_delete
        }
    elif isinstance(data, list):
        return [delete_keys_recursive(item, keys_to_delete) for item in data]
    return data

def prepare_schema_for_llm(model: type[BaseModel]) -> Dict[str, Any]:
    """
    Prepare a Pydantic model's JSON schema for use in LLM prompts, removing unnecessary fields and resolving references.
    
    Args:
        model (type[BaseModel]): Pydantic model class to generate the schema for.
    
    Returns:
        Dict[str, Any]: Processed JSON schema suitable for LLM integration.
    
    Raises:
        ValueError: If the schema generation fails or the model is invalid.
    """
    logger.info(f"Preparing schema for Pydantic model: {model.__name__}")
    try:
        # Generate the base JSON schema
        schema = model.model_json_schema()
        
        # Resolve any $ref references
        processed_schema = replace_value_in_dict(schema.copy(), schema.copy())
        
        # Remove unnecessary keys for LLM compatibility (e.g., $defs, title, default)
        keys_to_delete = ["$defs", "title", "default", "examples"]
        final_schema = delete_keys_recursive(processed_schema, keys_to_delete)
        
        # Validate schema structure
        if not isinstance(final_schema, dict) or "type" not in final_schema:
            raise ValueError(f"Invalid schema structure for model {model.__name__}")
        
        logger.info(f"Successfully prepared schema for {model.__name__}")
        return final_schema
    except Exception as e:
        logger.error(f"Failed to prepare schema for {model.__name__}: {e}")
        raise

# Example usage for testing
if __name__ == "__main__":
    # Example Pydantic model for testing
    class ExampleModel(BaseModel):
        name: str
        age: Optional[int] = None
        data: Dict[str, Any] = {}

    try:
        schema = prepare_schema_for_llm(ExampleModel)
        logger.info(f"Generated schema: {json.dumps(schema, indent=2)}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
