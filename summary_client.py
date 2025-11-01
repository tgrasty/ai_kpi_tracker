import os
from typing import Optional

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Optional: programmatic Hugging Face login using env token
def _login_huggingface_if_token_present() -> None:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return
    try:
        # Newer API
        from huggingface_hub import login as hf_login

        hf_login(token=token, add_to_git_credential=False)
        return
    except Exception:
        pass
    try:
        # Fallback for older versions
        from huggingface_hub.hf_api import HfFolder  # type: ignore

        HfFolder.save_token(token)
    except Exception:
        # If login fails, pipeline may still work if env var is respected by hub
        pass


class SummaryClient:
    """Base interface for summary generation."""
    def summarize(self, text: str) -> str:
        raise NotImplementedError


class GemmaSummaryClient(SummaryClient):
    """Uses Gemma (Google DeepMind) model locally or via Hugging Face."""
    def __init__(self, model_name: str = "distilgpt2"):
        if pipeline is None:
            raise ImportError("transformers not installed. Run `pip install transformers torch`")
        _login_huggingface_if_token_present()
        # Prefer CPU by default to avoid VRAM offload errors; use GPU if available
        try:
            import torch  # type: ignore
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False

        # Prefer 8-bit loading on CUDA if bitsandbytes is available
        use_8bit = False
        if has_cuda:
            try:
                import bitsandbytes as bnb  # type: ignore
                use_8bit = True
            except Exception:
                use_8bit = False

        def build_pipeline(model_kwargs):
            return pipeline(
                task="text-generation",
                model=model_name,
                model_kwargs=model_kwargs,
            )

        if use_8bit:
            try:
                model_kwargs = {
                    "device_map": "auto",
                    "load_in_8bit": True,
                    "low_cpu_mem_usage": True,
                }
                self.generator = build_pipeline(model_kwargs)
                return
            except Exception:
                # Fall back to CPU float32
                pass

        # CPU fallback (or CUDA without bitsandbytes)
        model_kwargs = {"device_map": "cpu", "low_cpu_mem_usage": True}
        try:
            import torch  # type: ignore
            model_kwargs["torch_dtype"] = torch.float32
        except Exception:
            pass
        self.generator = build_pipeline(model_kwargs)

    def summarize(self, text: str) -> str:
        # Use the prompt directly from app.py without modification
        # The text already contains the full prompt with questions and data
        # Increase max_new_tokens for comprehensive analysis (your prompt asks for detailed answers)
        try:
            output = self.generator(text, max_new_tokens=500, temperature=0.7, do_sample=True, return_full_text=False)
        except Exception as e:
            return f"Error generating summary: {str(e)}. The model may not be suitable for this task."
        
        # Debug: Check what was returned
        try:
            print(f"DEBUG Gemma - Output type: {type(output)}")
            print(f"DEBUG Gemma - Output: {output}")
            
            if not output:
                return "Error: No output generated from model"
            
            # Handle different output formats
            if isinstance(output, list):
                if len(output) == 0:
                    return "Error: Empty output from model"
                # Try to get generated_text from first item
                if isinstance(output[0], dict):
                    generated_text = output[0].get("generated_text", "")
                elif isinstance(output[0], str):
                    generated_text = output[0]
                else:
                    # Try to convert to string
                    generated_text = str(output[0])
            elif isinstance(output, dict):
                generated_text = output.get("generated_text", str(output))
            elif isinstance(output, str):
                generated_text = output
            else:
                generated_text = str(output)
            
            # Debug: Log what we got
            print(f"DEBUG Gemma - Original text length: {len(text)}")
            print(f"DEBUG Gemma - Generated text length: {len(generated_text)}")
            
            # Extract the generated text (return everything after the original prompt)
            if generated_text.startswith(text):
                result = generated_text[len(text):].strip()
                print(f"DEBUG Gemma - Extracted result length: {len(result)}")
                # If result is empty, the model just repeated the input
                if not result:
                    return "Error: Model did not generate new content. The model may not be suitable for this task, or try a different model."
                return result
            else:
                # Model might have reformatted or started differently
                # Return the full generated text
                result = generated_text.strip()
                print(f"DEBUG Gemma - Full result length: {len(result)}")
                return result
                
        except (IndexError, KeyError, TypeError) as e:
            return f"Error processing model output: {str(e)}. Output structure: {type(output)}. Try using OpenAI instead."


class OpenAISummaryClient(SummaryClient):
    """Optional: Uses OpenAI's GPT models if API key is provided."""
    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run `pip install openai`")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def summarize(self, text: str) -> str:
        # Use the prompt directly from app.py without modification
        # The text already contains the full prompt with questions and data
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content.strip()