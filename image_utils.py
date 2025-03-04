import base64
import io
import math
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from document_ai_agents.logger import logger
from typing import Optional
import asyncio

def pil_image_to_base64_jpeg(rgb_image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.
    
    Args:
        rgb_image (Image.Image): PIL Image object in RGB mode.
    
    Returns:
        str: Base64-encoded JPEG string.
    
    Raises:
        ValueError: If the image conversion fails.
        Exception: If encoding or saving fails.
    """
    logger.info("Converting PIL Image to base64 JPEG")
    try:
        buffered = io.BytesIO()
        rgb_image.save(buffered, format="JPEG", quality=85)  # Use quality=85 for balance between size and quality
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info(f"Successfully converted image to base64 JPEG, length: {len(img_str)}")
        return img_str
    except Exception as e:
        logger.error(f"Failed to convert PIL Image to base64 JPEG: {e}")
        raise

def image_file_to_base64_jpeg(image_path: str) -> str:
    """
    Convert an image file to a base64-encoded JPEG string.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Base64-encoded JPEG string.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image format is unsupported.
        Exception: If file reading or conversion fails.
    """
    logger.info(f"Converting image file to base64 JPEG: {image_path}")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        rgb_image = Image.open(image_path).convert("RGB")
        return pil_image_to_base64_jpeg(rgb_image)
    except Exception as e:
        logger.error(f"Failed to convert image file to base64 JPEG: {e}")
        raise

def base64_to_pil_image(base64_string: str) -> Image.Image:
    """
    Convert a base64-encoded string to a PIL Image.
    
    Args:
        base64_string (str): Base64-encoded JPEG string.
    
    Returns:
        Image.Image: PIL Image object.
    
    Raises:
        ValueError: If the base64 string is invalid or decoding fails.
        Exception: If image loading fails.
    """
    logger.info("Converting base64 string to PIL Image")
    try:
        image_data = base64.b64decode(base64_string)
        image_stream = io.BytesIO(image_data)
        pil_image = Image.open(image_stream).convert("RGB")
        logger.info("Successfully converted base64 to PIL Image")
        return pil_image
    except Exception as e:
        logger.error(f"Failed to convert base64 to PIL Image: {e}")
        raise

def draw_bounding_box_on_image(
    image: Image.Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    color: str = "red",
    thickness: int = 4,
    display_str_list: tuple = (),
    use_normalized_coordinates: bool = True,
) -> Image.Image:
    """
    Draw a bounding box and optional text on a PIL Image.
    
    Args:
        image (Image.Image): PIL Image to draw on.
        ymin (float): Minimum y-coordinate (normalized or absolute).
        xmin (float): Minimum x-coordinate (normalized or absolute).
        ymax (float): Maximum y-coordinate (normalized or absolute).
        xmax (float): Maximum x-coordinate (normalized or absolute).
        color (str): Color of the bounding box and text background (default: "red").
        thickness (int): Thickness of the bounding box line (default: 4).
        display_str_list (tuple): List of strings to display near the box (default: empty).
        use_normalized_coordinates (bool): If True, coordinates are normalized [0,1]; otherwise, absolute pixels.
    
    Returns:
        Image.Image: Modified PIL Image with bounding box and text.
    
    Raises:
        ValueError: If coordinates or image dimensions are invalid.
    """
    logger.info("Drawing bounding box on image")
    try:
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (
                xmin * im_width,
                xmax * im_width,
                ymin * im_height,
                ymax * im_height,
            )
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        
        # Validate coordinates
        if not (0 <= left < right <= im_width and 0 <= top < bottom <= im_height):
            raise ValueError("Invalid coordinates for bounding box")

        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
            width=thickness,
            fill=color,
        )
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        text_bottom = top
        for display_str in display_str_list[::-1]:
            _, _, text_width, text_height = font.getbbox(display_str)
            margin = math.ceil(0.05 * text_height)
            draw.rectangle(
                [
                    (left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom),
                ],
                fill=color,
            )
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font,
            )
            text_bottom -= text_height - 2 * margin
        logger.info("Successfully drew bounding box on image")
        return image
    except Exception as e:
        logger.error(f"Failed to draw bounding box on image: {e}")
        raise

async def async_pil_image_to_base64_jpeg(rgb_image: Image.Image) -> str:
    """
    Asynchronously convert a PIL Image to a base64-encoded JPEG string using a thread pool.
    
    Args:
        rgb_image (Image.Image): PIL Image object in RGB mode.
    
    Returns:
        str: Base64-encoded JPEG string.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, pil_image_to_base64_jpeg, rgb_image)

async def async_image_file_to_base64_jpeg(image_path: str) -> str:
    """
    Asynchronously convert an image file to a base64-encoded JPEG string using a thread pool.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Base64-encoded JPEG string.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, image_file_to_base64_jpeg, image_path)

async def async_base64_to_pil_image(base64_string: str) -> Image.Image:
    """
    Asynchronously convert a base64-encoded string to a PIL Image using a thread pool.
    
    Args:
        base64_string (str): Base64-encoded JPEG string.
    
    Returns:
        Image.Image: PIL Image object.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, base64_to_pil_image, base64_string)

async def async_draw_bounding_box_on_image(
    image: Image.Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    color: str = "red",
    thickness: int = 4,
    display_str_list: tuple = (),
    use_normalized_coordinates: bool = True,
) -> Image.Image:
    """
    Asynchronously draw a bounding box and optional text on a PIL Image using a thread pool.
    
    Args:
        image (Image.Image): PIL Image to draw on.
        ymin (float): Minimum y-coordinate (normalized or absolute).
        xmin (float): Minimum x-coordinate (normalized or absolute).
        ymax (float): Maximum y-coordinate (normalized or absolute).
        xmax (float): Maximum x-coordinate (normalized or absolute).
        color (str): Color of the bounding box and text background (default: "red").
        thickness (int): Thickness of the bounding box line (default: 4).
        display_str_list (tuple): List of strings to display near the box (default: empty).
        use_normalized_coordinates (bool): If True, coordinates are normalized [0,1]; otherwise, absolute pixels.
    
    Returns:
        Image.Image: Modified PIL Image with bounding box and text.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        draw_bounding_box_on_image,
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        thickness,
        display_str_list,
        use_normalized_coordinates,
    )
