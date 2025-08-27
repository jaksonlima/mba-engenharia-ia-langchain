from langchain.prompts import PromptTemplate
#from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text ="""
LangChain is a framework for 
 developing applications powered 
 by language models. It can be used for 
 chatbots, Generative Question-Answering 
 (GQA), summarization, and much more.
 """


splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=70)

parts = splitter.create_documents([long_text])

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

chain_sumarize = load_summarize_chain(llm, chain_type="stuff", verbose=False)

result = chain_sumarize.invoke({"input_documents": parts})

print(result)