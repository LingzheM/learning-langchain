from langchain_core.indexing import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

connection = ""
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"


vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url=connection
)

record_manager.create_schema()



docs = [

    Document(page_content='there are cats in the pond', metadata={
            "id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={
            "id": 2, "source": "ducks.txt"}),
]


index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # prevent duplicate documents
    source_id_key="source"  # use the source field as the source_id
)

print("Index attempt 1:", index_1)

index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source"
)

print("Index attempt 2:", index_2)

# If we mutate a document, the new version will be written and all old versions sharing the same source will be deleted.

docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)