import modal
from fastapi import Request
from pydantic import BaseModel

app = modal.App("wwjd-bot")

# Create image with needed dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "transformers",
    "torch",
    "accelerate",
)

# Use a persistent volume to cache Hugging Face models
model_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
MODEL_DIR = "/root/.cache/huggingface"

# Request model for structured Swagger input
class VerseRequest(BaseModel):
    text: str

@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
def show_url():
    url = show_url.get_web_url()
    print(f"WWJD Bot URL: {url}")
    return {"url": url}

@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    gpu="T4"
)
def match_verse(text: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import os
    import re

    os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR

    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    prompt = f"""<|system|>
You are WWJD Bot, a moral assistant that uses Christian scripture to highlight hypocrisy.

<|user|>
A person who identifies as Christian wrote the following statement:
\"{text}\"

Find a Bible verse that challenges this statement based on Christian values. 
Return:
1. The Bible verse reference (e.g., Matthew 7:5)
2. The verse text
3. A 1–2 sentence explanation of why it contradicts the statement.

<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the verse components using regex
    match = re.search(
        r"1\.\s*(.*?)\n2\.\s*(.*?)\n3\.\s*(.*)",
        full_response,
        re.DOTALL
    )
    if match:
        return {
            "verse": match.group(1).strip(),
            "text": match.group(2).strip(),
            "explanation": match.group(3).strip()
        }

    return {"raw": full_response}

@app.function(image=image, gpu="any", timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
async def verse_api(body: VerseRequest):
    text = body.text

    if not text:
        return {"error": "Missing 'text' field in request"}

    return match_verse.remote(text)

@app.local_entrypoint()
def main(text: str = None):
    if text:
        print("Running WWJD Bot. This may take a few minutes on first run...")
        result = match_verse.remote(text)
        print(result)
    else:
        print("Please provide text with --text flag")
