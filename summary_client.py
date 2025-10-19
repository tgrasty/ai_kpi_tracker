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
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
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
        prompt = f"Summarize this business performance data in one short, executive-style paragraph:\n\n{text}\n\nSummary:"
        output = self.generator(prompt, max_new_tokens=100, temperature=0.5)
        return output[0]["generated_text"].split("Summary:")[-1].strip()


class OpenAISummaryClient(SummaryClient):
    """Optional: Uses OpenAI's GPT models if API key is provided."""
    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run `pip install openai`")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def summarize(self, text: str) -> str:
        prompt = f"Summarize the business performance in plain English: {text}"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()