from dotenv import load_dotenv
load_dotenv()


import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipline = transformers.pipeline(
    task="text-generation",
    model=model_id,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16}
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipline(messages, max_new_tokens=100)


print(outputs[0]["generated_text"][-1])