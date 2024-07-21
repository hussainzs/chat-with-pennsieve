import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm

# Load environment variables
load_dotenv()

# Initialize Neo4jGraph using environment variables
graph = setup_neo4j_graph()
paths = format_paths_for_llm(extract_dataguide_paths(graph))
print(paths)






