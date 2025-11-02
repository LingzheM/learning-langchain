from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

connection = ""

raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)


embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(documents=documents, embeddings_model=embeddings_model, connection=connection)

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 2})

query = 'Who are the key figures in the ancient greek history of philosophy ?'

# fetch relevant documents
docs = retriever.invoke(query)
print(docs[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question}"""
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm_chain = prompt | llm

# answer the question based on the relevant documents
result = llm_chain.invoke({"context": docs, "question": query})

print(result)
print("\n\n")


print("Running again but this time encapsulate the logic for efficiency\n")

@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

result = qa.invoke(query)
print(result.content)