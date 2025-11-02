from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./test.pdf')
pages = loader.load()

print(pages)