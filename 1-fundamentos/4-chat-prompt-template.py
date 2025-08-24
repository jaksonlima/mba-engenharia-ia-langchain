from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

system = ("system", "you area an assistant that answers questions in a {style} style")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="funny", question="Who is Allan Turing")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

# model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
# result = model.invoke(messages)

gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
result = gemini.invoke(messages)

print(result.content)