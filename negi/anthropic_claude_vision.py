import os
import base64
import anthropic
import torch
import torchvision

class AnthropicClaudeVision:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        self.__client = anthropic.Client(api_key=api_key)
        self.__tmp_file = "claude_vision_tmp.jpg"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "model": ([
                    "claude-3-5-sonnet-20241022"
                ], {"default": "claude-3-5-sonnet-20241022"}),
                "detail": (["auto", "low", "high"],),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What's in this image?"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"
    OUTPUT_NODE = False
    CATEGORY = "Generator"

    def doit(self, image, seed, model, detail, max_tokens, prompt):
        _ = seed  # Unused but kept for interface compatibility
        _ = detail  # Unused but kept for interface compatibility

        # Convert image to PIL and save temporarily
        im0 = torchvision.transforms.functional.to_pil_image(torch.permute(image[0], (2, 0, 1)))
        im0.save(self.__tmp_file)
        
        # Read and encode image
        with open(self.__tmp_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            message = self.__client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Extract text from response
            response_text = ""
            for content_block in message.content:
                if content_block.type == "text":
                    response_text += content_block.text
            
            return (response_text,)
            
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
        finally:
            # Cleanup temporary file
            if os.path.exists(self.__tmp_file):
                os.remove(self.__tmp_file)