import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

res = client.completions.create(
    model="llama3.1",
    temperature=0.5,
    max_tokens=100,
    prompt="请给我的花店起个名字，用中文回复我"
)

print(res)