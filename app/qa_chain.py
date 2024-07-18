import os
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm
from prompt_generator import get_cypher_prompt_template

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize Neo4jGraph using environment variables
graph = setup_neo4j_graph()

# Extract DataGuide paths
results = extract_dataguide_paths(graph)
formatted_paths = "\n".join(format_paths_for_llm(results))


# Initialize ChatOpenAI with API key and model
llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.5,
    model="gpt-4"
)

# Refresh schema to ensure it's up-to-date
graph.refresh_schema()

# Get Cypher prompt template
chat_prompt = get_cypher_prompt_template()

# Create a partial prompt with schema and dataguide_paths filled in. user_query will be filled in later from user query.
partial_prompt = chat_prompt.partial(
    schema=graph.schema,
    dataguide_paths=formatted_paths
)

# Create the GraphCypherQAChain with the prompt, LLM, and graph
chain = GraphCypherQAChain.from_llm(
    cypher_prompt=partial_prompt,
    llm=llm,
    graph=graph,
    verbose=True,
    validate_query=True,
    include_run_info=True
)

# Example usage
def run_query(query):
    response = chain.invoke(query)
    return response

