import asyncio
import os
from asyncio import wait_for

import dotenv
from langchain_community.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader, UnstructuredURLLoader, \
    SeleniumURLLoader, PlaywrightURLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_experimental.text_splitter import SemanticChunker

from config import EMBEDDING_MODEL, PG_COLLECTION_NAME
from scraper.urls import urls

dotenv.load_dotenv()
url = ["https://www.waterwiseplants.org/find-a-plant/prickly-thrift/#header",
       "https://www.waterwiseplants.org/find-a-plant/cockspur-hawthorn/#header",
       "https://www.waterwiseplants.org/find-a-plant/cream-beauty-snow-crocus/#header"]


async def read_url():
    # Load HTML
    # loader = SeleniumURLLoader(url)
    # loader = AsyncChromiumLoader(url)
    doc = await loader.aload()
    # print(doc)
    return doc

# loop = asyncio.get_event_loop()
# docs = loop.run_until_complete(wait_for(read_url(), 2000))
# loop.close()

loader = PlaywrightURLLoader(urls=url, remove_selectors=["header", "footer"])
docs = loader.load()

# Transform
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(docs)
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
# transformer = UnstructuredURLLoader
# print(docs_transformed)
# loader = UnstructuredURLLoader(
#     urls=url, mode="elements", strategy="fast",
# )
# docs = loader.load()
# # print(docs)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

text_splitter = SemanticChunker(
    embeddings=embeddings
)
#
chunks = text_splitter.split_documents(docs_transformed)
print(chunks)

# Result
# print(len(docs_transformed))
# print(docs_transformed[0].metadata["source"])

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=PG_COLLECTION_NAME,
    connection_string=os.getenv("POSTGRES_URL"),
    pre_delete_collection=True
)


