from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    """
    An answer to the user's question along with justification for the answer.
    """
    answer: str

    justification: str


llm = ChatOpenAI(model="gpt-3.5", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke(
    "What weights more, a pound of bricks or pound of feathers"
)
print(response)