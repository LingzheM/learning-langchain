from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

connecton = ""
collection_name = ""
embeddings_model = OpenAIEmbeddings()
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()

print("length of loaded docs: ", len(docs[0].page_content))

# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


prompt_text = "Summarize the following document:\n\n{doc}"

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summarize_chain = {
    "doc": lambda x: x.page_content
} | prompt | llm | StrOutputParser()

summarizes = summarize_chain.batch(chunks, {"max_concurrency": 5})

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connecton,
    use_jsonb=True
)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in chunks]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summarizes)
]

retriever.vectorstore.add_documents(summary_docs)

# Store the original documents in the document store, linked to their summaries via doc_ids
# This allows us to first search summaries efficiently, then fetch the full docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))

sub_docs = retriever.vectorstore.similarity_search(
    "chapter on philosophy", k=2
)

print("sub docs: ", sub_docs[0].page_content)

print("length of sub docs:\n", len(sub_docs[0].page_content))


# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("chapter on philosophy")

print("length of retrieved docs: ", len(retrieved_docs[0].page_content))