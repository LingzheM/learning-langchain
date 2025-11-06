from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOpenAI()

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=MemorySaver())

thread1 = {"configurable": {"thread_id": "1"}}

result_1 = graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)
print(result_1)

result_2 = graph.invoke({"messages": [HumanMessage("What is my name ?")]}, thread1)
print(result_2)

print(graph.get_state(thread1))