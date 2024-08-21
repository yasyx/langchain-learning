from dotenv import load_dotenv

load_dotenv()

import  requests
from PIL import Image
from transformers import BlipProcessor ,BlipForConditionalGeneration
from langchain.tools import BaseTool
from  langchain_openai import OpenAI
from  langchain.agents import initialize_agent, AgentType


#---- Part I 初始化图像字幕生成模型
# 指定要使用的工具模型（HuggingFace中的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

processor = BlipProcessor.from_pretrained(hf_model)

model = BlipForConditionalGeneration.from_pretrained(hf_model)


class ImageCapTool(BaseTool):
    name = "Image captioner"
    description = "为图片创作说明文案"

    def _run(self,url: str):
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption


    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


llm = OpenAI(temperature=0.2)


tools = [ImageCapTool()]
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)
# 未来需要改成下面的设计
# agent = create_react_agent(
#     tools=tools,
#     llm=llm,
# )
img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'

input_dict = {"input": f"{img_url}\n请创作合适的中文推广文案"}

agent.invoke(input_dict)