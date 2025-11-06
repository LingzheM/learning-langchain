from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
# replace this with the connection details of your db
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# excute sql query
execute_query = QuerySQLDatabaseTool(db=db)

# combined_chain = write_query | execute_query
combined_chain = write_query | execute_query

result = combined_chain.invoke({"question": "how many employeess are there ?"})

print(result)