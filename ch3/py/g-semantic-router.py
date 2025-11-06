from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



pyhsics_template = """You are a very smart physics professor."""
math_template = """s"""

embeddings = OpenAIEmbeddings()
prompt_templates = [pyhsics_template, math_template]
prompt_embeddingss = embeddings.embed_documents(prompt_templates)


# Route question to prompt
@chain
def prompt_router(query):
    query_embedding = embeddings.embed_query(query)
    similarity = cosine_similarity([query_embedding], prompt_embeddingss)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

semantic_router = (prompt_router | ChatOpenAI() | StrOutputParser())

result = semantic_router.invoke("what's a black hole")
print("\nSemantic router result: ", result)

