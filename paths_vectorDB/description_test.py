import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

content_str = """You are a Neo4j expert and given a cypher path you will provide a description of what it means. 
Some info about the underlying graph structure and what it means:
0. Our database is graph database with medical datasets from Pennsieve. It contains all sorts of files within different datasets.
1. All key-value data is represented with "key" as the edge and "value" as the node.
2. Arrays are represented with index as the edge and value at the index as the node.
You job is to read a cypher path, read its relationship labels and node labels and provide a brief description of what this path might mean as a user query. All paths will start from root named :Pennsieve.
Here are 2 examples:
Example query: MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset {name: "Test Dataset CNT"})-[:FILES]->(f:File {name: "banner.jpg"}) RETURN d.name, f.name
Example response2: This path queries file named "banner.jpg" in dataset named "Test Dataset CNT".
Example query: MATCH (p:Pennsieve)-[:DATASET]->()-[:FILES]->()-[:DATA]->()-[:contributors]->()-[:first_name]->(fn:Data)
Example response: This path queries the first names of contributors in our database.

In your response Only provide a short description of the path and nothing else at all.
"""

# Test OpenAI API connection
api_key = os.environ['OPENAI_API_KEY']
if api_key:
    llm = ChatOpenAI(api_key=api_key, temperature=0.5, model="gpt-4")
    test_prompt = "MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset) RETURN DISTINCT d.name AS dataset_name"
    print("\nTesting OpenAI API connection:")
    messages = [HumanMessage(content=test_prompt), SystemMessage(content=content_str)]
    response = llm.invoke(messages)
    print(response)
else:
    print("OpenAI API key is not set. Please check your .env file.")