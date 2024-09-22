import os

from dotenv import load_dotenv
load_dotenv()

from ollama import Client

client = Client(
    host="http://localhost:11434/v1/"
)

response = client.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response)