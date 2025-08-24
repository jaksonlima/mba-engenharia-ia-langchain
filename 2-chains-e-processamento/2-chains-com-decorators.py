from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

@chain
def square(input_dict: dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x * x}

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me abount the number {square_result}"
)

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

#chain = question_template | gemini
chain2 = square | question_template2 | gemini

#result = chain.invoke({"name": "Jakson Lima"})
result = chain2.invoke({"x": 10})


print(result.content)