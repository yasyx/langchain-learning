from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate


template = """ 您是一位专业的花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name}，您能够撰写一份吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)

print(prompt)


from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1")

inputMsg = prompt.format(price=100, flower_name=["玫瑰"])

output = model.invoke(inputMsg)

print(output)