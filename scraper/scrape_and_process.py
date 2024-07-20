import asyncio
from asyncio import wait_for

import dotenv
from langchain_community.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader, UnstructuredURLLoader, \
    SeleniumURLLoader, PlaywrightURLLoader
from langchain_community.document_transformers import BeautifulSoupTransformer, Html2TextTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from config import EMBEDDING_MODEL
from scraper.urls import urls

dotenv.load_dotenv()


async def load(loader):
    return await loader.aload()


async def read_url():
    # Load HTML
    url = ["https://www.waterwiseplants.org/find-a-plant/prickly-thrift/#header"]
    # loader = SeleniumURLLoader(["https://www.waterwiseplants.org/find-a-plant/"])
    # loader = PlaywrightURLLoader(urls=url, remove_selectors=["header", "footer"])

    loader = AsyncChromiumLoader(url)

    doc = await load(loader)
    # print(doc)
    return doc


loop = asyncio.get_event_loop()
docs = loop.run_until_complete(wait_for(read_url(), 2000))
loop.close()

# Transform
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["div", "p"])
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings()
)

chunks = text_splitter.split_documents(docs)
print(chunks)

# Result
print(len(docs_transformed))
print(docs_transformed[0].page_content)
