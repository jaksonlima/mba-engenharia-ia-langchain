from langchain.prompts import PromptTemplate
#from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text to English: {text}",
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text 4 words: {text}",
)

llm_en = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"text": "languechain é um framework para desenvolvimento de aplicações com LLMs"})

print(result)