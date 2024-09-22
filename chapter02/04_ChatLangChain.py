from dotenv import load_dotenv

load_dotenv()

# from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="gpt-3.5-turbo",
    temperature=0.8,
    max_tokens=60
)

from langchain.schema import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Why is the sky blue?")
]

res = chat.invoke(messages)

print(res)