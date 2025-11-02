from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)

model = ChatOpenAI()

prompt = template.invoke(
    {
        "context": "The most recent advancementss in NLP are being driven by Large Language Models (LLMs).",
        "question": "Which model providers offer LLMs?",
    }
)


print(model.invoke(prompt))