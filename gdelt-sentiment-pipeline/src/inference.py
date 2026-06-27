"""
Batched generation loop: builds prompts, runs the model, and parses each
decoded output into a structured score dict.
"""
import torch
from tqdm.auto import tqdm

from config import BATCH_SIZE, MODEL_MAX_INPUT_TOKENS, MODEL_MAX_NEW_TOKENS
from src.llm_scoring import build_fingpt_prompt, parse_fingpt_response


def run_batched_inference(headlines, model, tokenizer, eos_ids):
    """
    Run the scoring prompt over all headlines in batches of BATCH_SIZE.

    Returns a list of parsed score dicts, one per headline, in the same
    order as the input list.
    """
    parsed_structs = []

    for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc="Llama 3 Structural Scoring Run"):
        batch_lines = headlines[i: i + BATCH_SIZE]
        prompts = [build_fingpt_prompt(h) for h in batch_lines]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MODEL_MAX_INPUT_TOKENS,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MODEL_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, out_tensor in enumerate(outputs):
            input_len = inputs["input_ids"][idx].shape[0]
            gen_tokens = out_tensor[input_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            result = parse_fingpt_response(decoded, headline=batch_lines[idx])
            parsed_structs.append(result)

    return parsed_structs
