# git clone https://github.com/nlpai-lab/KULLM.git
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from KULLM.utils.prompter import Prompter

MODEL = "nlpai-lab/kullm-polyglot-5.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir='/data/ephemeral/huggingface',
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


result = infer(input_text="고려대학교에 대해서 알려줘")
print(result)