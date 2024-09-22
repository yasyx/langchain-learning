from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate

template = """ 您是一位专业的花店文案撰写员。\n
对于售价为 {price} 元的 {flower_name}，您能够撰写一份吸引人的简短描述吗？
{format_instructions}"""

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="reason", description="问什么要这样写这个文案"),
    ResponseSchema(name="description", description="鲜花的描述文案"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate.from_template(template, partial_variables={"format_instructions": format_instructions})

print(prompt)

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.1")

# 创建一个空的DataFrame用于存储结果
import pandas as pd

df = pd.DataFrame(columns=["flower", "price", "description", "reason"])  # 先声明列名

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)

    # 获取模型的输出
    output = model.invoke(input)
    print(output)
    # 解析模型的输出（这是一个字典结构）
    parsed_output = output_parser.parse(output.content)

    # 在解析后的输出中添加“flower”和“price”
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将解析后的输出添加到DataFrame中
    df.loc[len(df)] = parsed_output

# 打印字典
print(df.to_dict(orient='records'))

# 保存DataFrame到CSV文件
df.to_csv("flowers_with_descriptions.csv", index=False)
