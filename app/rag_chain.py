import asyncio
import os
from asyncio import wait_for
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config import PG_COLLECTION_NAME

load_dotenv()


class RagInput(TypedDict):
    question: str


vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string=os.getenv("POSTGRES_URL"),
    embedding_function=OpenAIEmbeddings()
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-1106-preview",
    streaming=True
)

multiquery = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=llm)

no_history_chain = (
        RunnableParallel(
            context=(itemgetter("question") | multiquery),
            question=itemgetter("question")
        )
        | RunnableParallel(
            answer=(ANSWER_PROMPT | llm),
            docs=itemgetter("context")
        )
).with_types(input_type=RagInput)

# history_retriever = lambda session_id: SQLChatMessageHistory(
#     connection_string=os.getenv("POSTGRES_MEMORY_URL"),
#     session_id=session_id
# )

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its orignal language,
#
# Chat history:
# {chat_history}
# Follow Up Input: {question}
# Standalone Question:"""s
# condense_question_prompt = PromptTemplate.from_template(_template)

# standalone_question = RunnableParallel(
#     question=RunnableParallel(
#         question=RunnablePassthrough(),
#         chat_history=lambda x: get_buffer_string(x["chat_history"])
#     ) | condense_question_prompt | llm | StrOutputParser()
# )

# final_chain = RunnableWithMessageHistory(
#     runnable= no_history_chain,
#     input_messages_key="question",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
#     get_session_history=history_retriever
# )

# NO_HISTORY_CHAIN_INVOKE = no_history_chain.invoke({"question": "What is a low water, full sun plant you recommend?"})
# print(NO_HISTORY_CHAIN_INVOKE)


