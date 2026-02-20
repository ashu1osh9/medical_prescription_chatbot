"""
Vision API wrapper using Google Gemini for multimodal model integration.
Handles streaming communication with Gemini Vision API.
"""
import os
import base64
from typing import Dict, List, Any, Iterator
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class VisionLLMClient:
    """
    Gemini-based wrapper for vision model.
    Drop-in replacement for the Qubrid client.
    """

    def __init__(self):
        self.api_key = os.getenv("VISION_API_KEY")
        self.model_name = "gemini-1.5-flash"

        if not self.api_key:
            raise ValueError("VISION_API_KEY must be set in .env file")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream tokens from Gemini API.
        Converts OpenAI-format messages to Gemini format.
        """
        # Extract system prompt
        system_prompt = ""
        gemini_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Collect system prompt as text
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            system_prompt += part["text"] + "\n"
                else:
                    system_prompt += str(content) + "\n"

            elif role == "user":
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            gemini_parts.append(part["text"])
                        elif part.get("type") == "image_url":
                            image_url = part["image_url"]["url"]
                            # Handle base64 data URI
                            if image_url.startswith("data:image"):
                                header, b64data = image_url.split(",", 1)
                                mime_type = header.split(";")[0].split(":")[1]
                                image_bytes = base64.b64decode(b64data)
                                gemini_parts.append({
                                    "mime_type": mime_type,
                                    "data": image_bytes
                                })
                else:
                    gemini_parts.append(str(content))

            elif role == "assistant":
                # Skip assistant messages for now (single turn)
                pass

        # Prepend system prompt to first text part
        if system_prompt:
            gemini_parts.insert(0, system_prompt)

        # Build Gemini-format contents
        gemini_contents = []
        for part in gemini_parts:
            if isinstance(part, str):
                gemini_contents.append(part)
            elif isinstance(part, dict):
                # Image part
                import google.generativeai as genai_img
                gemini_contents.append(
                    genai_img.protos.Blob(
                        mime_type=part["mime_type"],
                        data=part["data"]
                    )
                )

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.model.generate_content(
            gemini_contents,
            generation_config=generation_config,
            stream=True
        )

        for chunk in response:
            try:
                if chunk.text:
                    yield chunk.text
            except Exception:
                continue
