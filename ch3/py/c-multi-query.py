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

db = PGVector.from_documents(documents, embeddings_model, connection)

retriever = db.as_retriever(search_kwargs={"k": 5})

perspectives_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo")


def parse_queries_output(message):
    return message.content.split('\n')

query_gen = perspectives_prompt | llm | parse_queries_output

def get_unique_union(document_lists):
    deduped_docs = {
        doc.page_content: doc for sublist in document_lists for doc in sublist
    }
    return list(deduped_docs.values())


retrieval_chain = query_gen | retriever.batch | get_unique_union

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the key figures in the ancient greek history of philosophy?"

@chain
def multi_query_qa(input):
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({"context": docs, "question": input})
    answer = llm.invoke(formatted)
    return answer

print("Running multi query qa\n")
result = multi_query_qa.invoke(query)
print(result.content)