from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate


template = """ 您是一位专业的花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name}，您能够撰写一份吸引人的简短描述吗？
"""

prompt = PromptTemplate.from_template(template)

print(prompt)


from langchain_community.llms import HuggingFaceHub

model = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 256})

flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["100", "200", "300"]

for flower, price in zip(flowers, prices):
    inputMsg = prompt.format(price=price, flower_name=flower)
    output = model.invoke(inputMsg)
    print(output)
