import os
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
import time

from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm
from prompt_generator import get_cypher_prompt_template
from paths_vectorDB.main import get_similar_paths_from_milvus

# Load environment variables
from dotenv import load_dotenv


def run_query(user_query: str):
    """
    Executes a user query against a Neo4j graph database and returns the response.

    This function sets up the Neo4j graph, extracts DataGuide paths, initializes the ChatOpenAI model,
    refreshes the graph schema, generates a Cypher prompt template, performs a vector similarity search
    using Milvus, and finally invokes the `GraphCypherQAChain` with the user query.

    Args:
        user_query (str): The query string provided by the user.

    Returns:
        dict: The response from the GraphCypherQAChain, including the query results and intermediate steps.
    """
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
        model="gpt-4o"
    )

    # Refresh schema to ensure it's up-to-date
    graph.refresh_schema()

    # Get Cypher prompt template
    chat_prompt = get_cypher_prompt_template()
    start_time = time.time()
    few_shot_examples = get_similar_paths_from_milvus(graph=graph, user_query=user_query, top_k=3)
    end_time = time.time()
    print(f"****Time taken to conduct vector similarity search in vector DB: {end_time - start_time:.2f} seconds")

    # Create a partial prompt with schema and dataguide_paths filled in. user_query will be filled in later from user query.
    partial_prompt = chat_prompt.partial(
        schema=graph.schema,
        example_queries=few_shot_examples,
        dataguide_paths=formatted_paths
    )

    # Create the GraphCypherQAChain with the prompt, LLM, and graph
    chain = GraphCypherQAChain.from_llm(
        cypher_prompt=partial_prompt,
        llm=llm,
        graph=graph,
        verbose=True,
        validate_query=True,
        include_run_info=True,
        return_intermediate_steps=True
    )

    # Finally, Invoke the chain with the user query
    response = chain.invoke(user_query)
    return response

