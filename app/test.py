import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm

# Load environment variables
load_dotenv()

# Initialize Neo4jGraph using environment variables
# graph = setup_neo4j_graph()
# paths = format_paths_for_llm(extract_dataguide_paths(graph))
# print(paths)

content = """Some info about the underlying graph structure and what it means:
1. All key-value data is represented with "key" as the edge and "value" as the node.
2. Arrays are represented with index as the edge and value at the index as the node.

"""


# Test OpenAI API connection
api_key = os.environ['OPENAI_API_KEY']
if api_key:
    llm = ChatOpenAI(api_key=api_key, temperature=0.5, model="gpt-4")
    test_prompt = "Say Hello, you're connected to openai"
    print("\nTesting OpenAI API connection with a simple prompt:")
    messages = [HumanMessage(content=test_prompt), SystemMessage(content="Respond exactly as prompted")]
    response = llm.invoke(messages)
    print(response)
else:
    print("OpenAI API key is not set. Please check your .env file.")




