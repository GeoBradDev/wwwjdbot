import modal
from pydantic import BaseModel
import time

app = modal.App("wwjd-bot")

# Create image with needed dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "transformers",
    "torch",
    "accelerate",
    "bitsandbytes",
)

# Use a persistent volume to cache Hugging Face models
model_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
MODEL_DIR = "/root/.cache/huggingface"


class VerseRequest(BaseModel):
    text: str


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def match_verse(text: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    import os
    import re
    import json

    os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    token = os.environ.get("HUGGINGFACE_TOKEN")
    from huggingface_hub import login
    login(token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=token
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()

    prompt = f"""<s>[INST] 
    Analyze the following statement from the perspective of Christian teachings:

    "{text}"

    Determine whether the statement aligns with or contradicts biblical values, especially those emphasized by Jesus in the New Testament.

    Respond with a concise, respectful comment that:
    1. Cites a specific Bible verse that applies
    2. Includes the full text of that verse
    3. Explains—briefly and clearly—how the verse relates to the statement

    The goal is to offer thoughtful moral insight, not judgment. The tone should be direct but compassionate, rooted in scripture and Christian principles of love, justice, and humility.

    Return the result in JSON format with three fields:
    {{
      "reference": "The Bible verse reference (e.g., Matthew 5:44)",
      "verse": "The full text of the verse",
      "explanation": "A short explanation (1–2 sentences) of how the verse relates to the original statement"
    }}
    [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            min_length=inputs.input_ids.shape[1] + 20,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    elapsed_time = time.time() - start_time

    print("=== Prompt Sent ===")
    print(prompt)
    print("=== Model Output ===")
    print(generated_text)
    print(f"Response generated in {elapsed_time:.2f} seconds")

    json_match = re.search(r'({[\s\S]*})', generated_text)

    try:
        if json_match:
            json_str = json_match.group(1)
            response_json = json.loads(json_str)

            if all(k in response_json for k in ['reference', 'verse', 'explanation']):
                return {
                    "reference": response_json["reference"],
                    "verse": response_json["verse"],
                    "explanation": response_json["explanation"],
                    "model": model_name,
                    "generation_time": f"{elapsed_time:.2f} seconds"
                }

        reference_pattern = re.compile(r'"reference"\s*:\s*"([^"]+)"')
        verse_pattern = re.compile(r'"verse"\s*:\s*"([^"]+)"')
        explanation_pattern = re.compile(r'"explanation"\s*:\s*"([^"]+)"')

        reference_match = reference_pattern.search(generated_text)
        verse_match = verse_pattern.search(generated_text)
        explanation_match = explanation_pattern.search(generated_text)

        reference = reference_match.group(1) if reference_match else ""
        verse = verse_match.group(1) if verse_match else ""
        explanation = explanation_match.group(1) if explanation_match else ""

        if reference or verse or explanation:
            return {
                "reference": reference,
                "verse": verse,
                "explanation": explanation,
                "model": model_name,
                "generation_time": f"{elapsed_time:.2f} seconds"
            }

        return {
            "reference": "",
            "verse": "",
            "explanation": generated_text,
            "model": model_name,
            "generation_time": f"{elapsed_time:.2f} seconds"
        }

    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}", "raw_output": generated_text}


@app.function(image=image, gpu="any", timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
async def verse_api(body: VerseRequest):
    text = body.text

    if not text:
        return {"error": "Missing 'text' field in request"}

    try:
        result = match_verse.remote(text)
        return {
            "raw": {
                "reference": result.get("reference", ""),
                "verse": result.get("verse", ""),
                "explanation": result.get("explanation", "")
            }
        }
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
