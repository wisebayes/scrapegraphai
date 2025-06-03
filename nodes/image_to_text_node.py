"""
ImageToTextNode Module
"""

from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage

from .base_node import BaseNode
import base64

import httpx
from langchain.chat_models import init_chat_model
import io
from PIL import Image



class ImageToTextNode(BaseNode):
    """
    Retrieve images from a list of URLs and return a description of
    the images using an image-to-text model.
    If an image has transparency, it's rendered on a black background before description.

    Attributes:
        llm_model: An instance of the language model client used for image-to-text conversion.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "ImageToText".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "ImageToText",
    ):
        super().__init__(node_name, "node", input, output, 1, node_config)

        self.llm_model = node_config["llm_model"]
        self.verbose = node_config.get("verbose", False)
        self.max_images = node_config.get("max_images", 5)

    def _get_mime_type_from_url_or_header(self, url: str, headers: Optional[httpx.Headers] = None) -> str:
        """Determines MIME type from headers or URL, with a fallback."""
        if headers:
            content_type = headers.get("Content-Type")
            if content_type and content_type.startswith("image/"):
                return content_type
        
        url_lower = url.lower()
        if "png" in url_lower:
            return "image/png"
        elif "jpg" in url_lower or "jpeg" in url_lower:
            return "image/jpeg"
        elif "gif" in url_lower:
            return "image/gif"
        elif "webp" in url_lower:
            return "image/webp"
        self.logger.warning(f"Could not determine MIME type for {url} from headers or URL, defaulting to image/jpeg.")
        return "image/jpeg"

    def _check_transparency(self, pil_image_rgba: Image.Image) -> bool:
        """Checks if an RGBA Pillow image has any non-opaque pixels."""
        # Ensure it's RGBA. If it was, for example, RGB, convert() would add a fully opaque alpha.
        # If it was L (greyscale) or P (palette), it also gets an alpha.
        # This check is against truly transparent or semi-transparent pixels.
        if pil_image_rgba.mode != 'RGBA':
            # This case should ideally not be hit if images are always converted to RGBA first.
            # If it's not RGBA, it cannot have an alpha channel in the typical sense.
            self.logger.debug(f"Image mode is {pil_image_rgba.mode}, not RGBA. Assuming opaque for transparency check.")
            return False

        # Check if min alpha is less than 255 (fully opaque)
        # .getextrema() returns (min, max) for a single band image (like the alpha channel)
        try:
            alpha_min, _ = pil_image_rgba.getchannel('A').getextrema()
            return alpha_min < 255
        except Exception as e:
            self.logger.warning(f"Could not get alpha channel for transparency check: {e}. Assuming opaque.")
            return False

    def _create_image_variants_from_url(self, url: str) -> Tuple[Optional[str], Optional[str], str, bool]:
        """
        Fetches an image. If transparent, creates black/white bg variants (returns black for LLM).
        If opaque, returns original.

        Returns:
            Tuple: (
                image_data_for_llm (str, base64, or None if error),
                image_data_white_variant (str, base64, or None if not processed or error),
                mime_type_for_llm (str),
                was_transparent_and_processed (bool)
            )
        """
        image_data_for_llm = None
        image_data_white_bg = None # Only populated if transparent and processed
        mime_type = "application/octet-stream" # Default in case of early error
        was_processed = False

        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()
            img_bytes = response.content
            headers = response.headers
            
            original_pil_image = Image.open(io.BytesIO(img_bytes))
            # Convert to RGBA for consistent transparency checking and processing
            pil_image_rgba = original_pil_image.convert("RGBA")

            if self._check_transparency(pil_image_rgba):
                if self.verbose:
                    self.logger.info(f"Image {url} has transparency. Processing variants.")
                was_processed = True
                mime_type = "image/png" # Processed images are saved as PNG

                # Create black background version
                black_bg_img = Image.new("RGBA", pil_image_rgba.size, "BLACK")
                black_bg_img.paste(pil_image_rgba, (0, 0), pil_image_rgba)
                buffer_black = io.BytesIO()
                black_bg_img.convert("RGB").save(buffer_black, format="PNG")
                image_data_for_llm = base64.b64encode(buffer_black.getvalue()).decode("utf-8")

                # Create white background version
                white_bg_img = Image.new("RGBA", pil_image_rgba.size, "WHITE")
                white_bg_img.paste(pil_image_rgba, (0, 0), pil_image_rgba)
                buffer_white = io.BytesIO()
                white_bg_img.convert("RGB").save(buffer_white, format="PNG")
                image_data_white_bg = base64.b64encode(buffer_white.getvalue()).decode("utf-8")
            else:
                if self.verbose:
                    self.logger.info(f"Image {url} is opaque. Using original.")
                was_processed = False
                image_data_for_llm = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = self._get_mime_type_from_url_or_header(url, headers)

        except httpx.RequestError as e:
            self.logger.error(f"HTTP request error fetching image from {url}: {e}")
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP status error fetching image from {url}: {e}")
        except IOError as e:  # PIL specific errors
            self.logger.error(f"Pillow error processing image from {url}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while processing image {url}: {e}")

        return image_data_for_llm, image_data_white_bg, mime_type, was_processed

    def execute(self, state: dict) -> dict:
        """
        Generate text from an image using an image-to-text model. The method retrieves the image
        from the list of URLs provided in the state and returns the extracted text.

        Args:
            state (dict): The current state of the graph. The input keys will be used to fetch the
                            correct data types from the state.

        Returns:
            dict: The updated state with the input key containing the text extracted from the image.
        """

        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        urls = input_data[0]

        if not urls:
            self.logger.info("URLs list is empty, skipping image processing.")
            state.update({self.output[0]: []})
            return state

        if self.max_images < 1:
            self.logger.info("max_images is less than 1, skipping image processing.")
            state.update({self.output[0]: []})
            return state
        
        urls_to_process = urls[: self.max_images]
        img_desc = []

        for url in urls_to_process:
            image_data_for_llm, _image_data_white_variant, mime_type, was_processed = self._create_image_variants_from_url(url)

            text_answer = "Error: Failed to process image or model failure." # Default error

            if image_data_for_llm:
                prompt_text = "Describe the provided image."
                if was_processed:
                    print(was_processed)
                    prompt_text = "Describe the image: rendered as 2 copies due to transparent background (Transparent areas are rendered on a black background), (Transparent areas are rendered on a white background)."
                
                try:
                    # Langchain's HumanMessage with Anthropic-style image content format:
                    print(prompt_text)
                    message_content = [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type, 
                                "data": image_data_for_llm,
                            }
                        }
                    ]
                    if was_processed:
                        message_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type, 
                                "data": _image_data_white_variant,
                            }
                        })
                    message = HumanMessage(content=message_content)
                    
                    llm_response = self.llm_model.invoke([message])
                    text_answer = llm_response.content if llm_response else "Error: No content from model."
                    
                    if self.verbose:
                        self.logger.info(f"Image URL: {url}")
                        self.logger.info(f"Processed for transparency: {was_processed}")
                        self.logger.info(f"MIME type for LLM: {mime_type}")
                        self.logger.info(f"LLM Response: {text_answer}")

                except Exception as e:
                    self.logger.error(f"LLM invocation failed for {url}: {e}")
                    text_answer = "Error: Model failure during image description."
            else:
                self.logger.warning(f"Failed to create image variants for URL: {url}")
                # text_answer remains the default error set above
            
            img_desc.append(text_answer)

        state.update({self.output[0]: img_desc})
        return state
