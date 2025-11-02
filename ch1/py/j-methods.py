from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

completion = model.invoke("Hi there!")

completions = model.batch(["Hi there!", "Bye!"])

for token in model.stream("Bye!"):
    print(token)