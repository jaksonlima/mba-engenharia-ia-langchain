from langchain.tools import tool
#from langchain_openiai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

@tool("Calculadora", return_direct=True)
def calculator(expression: str) -> str:
    """Calcula uma expressão matemática simples."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Erro ao calcular a expressão: {e}"

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mocked web search."""
    data = {"Brazil": "Brasil", "France": "França", "Germany": "Alemanha"}
    for key, value in data.items():
        if key.lower() in query.lower():
            return f"{key} em português é {value}."
    return "Desculpe, não encontrei a informação."


llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0, disable_streaming=True)

tools = [calculator, web_search_mock]

prompt = PromptTemplate.from_template(
"""Você é um assistente útil. Você tem acesso às seguintes ferramentas:
{tools}

Os nomes das ferramentas que você pode chamar são:
{tool_names}

Instruções (mantenha os marcadores em inglês para o agente funcionar):
Use o seguinte formato:
Question: a pergunta do usuário
Thought: você deve pensar sobre o que fazer
Action: o NOME exato da ferramenta a ser usada, escolhido dentre {tool_names}
Action Input: a entrada para a ação (em JSON ou string)
Observation: o resultado da ação
... (repita Thought/Action/Action Input/Observation conforme necessário)
Thought: I now know the final answer
Final Answer: a resposta final ao usuário (em português)

Question: {input}
{agent_scratchpad}"""
)

agent_chain = create_react_agent(llm, tools, prompt, stop_sequence=False)

agent = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3)

print(agent.invoke({"input": "Qual é a soma de 15 e 30? E como se diz Brazil em português?"}))
