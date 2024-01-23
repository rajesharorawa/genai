from langchain import SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import os
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
import uuid
import re
import redis

st.title("Talk to your Oracle EBS financial data using natural language")



os.environ['OPENAI_API_KEY'] = "your_key"
os.environ["SERPAPI_API_KEY"] = "your_key"

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

from sqlalchemy import create_engine

engine = create_engine(
    f'oracle+oracledb://:@',
    connect_args={
        "user": "username",
        "password": "password",
        "dsn": "adwchg_low",
        "config_dir": "your_dir",
        "wallet_location": "your_dir",
        "wallet_password": "password",
    })

db = SQLDatabase(
engine=engine,
schema="user",
include_tables=["your tables"],
)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True,return_direct=True)

dbtool = Tool(
        name="dbchain",
        func=db_chain.run,
        description="this tool is used to execute sql queries against the database or a schema",
    )

tools1 = load_tools(["serpapi"])
tools =  [dbtool] + tools1
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}


message_history = RedisChatMessageHistory(
    url="rediss://hostname:6379", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="memory", return_messages=True, chat_memory=message_history
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]

def drawgraph(text):

    if "plot" in text.lower() and "python" in text.lower() and "graph" in text.lower():
        code = extract_python_code(text)
        code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
        return code
    else:
        return None

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        graphcode = drawgraph(response)
        if graphcode:
            exec(graphcode)
        st.write(response)
