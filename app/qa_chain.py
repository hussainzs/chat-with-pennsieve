import os
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
import time

from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm
from app.prompt_generator import get_cypher_prompt_template
from paths_vectorDB.main import get_similar_paths_from_milvus

# Load environment variables
from dotenv import load_dotenv


def run_query(user_query: str, max_retries: int = 3):
    """
    Executes a user query against a Neo4j graph database and returns the response.

    This function sets up the Neo4j graph, extracts DataGuide paths, initializes the ChatOpenAI model,
    refreshes the graph schema, generates a Cypher prompt template, performs a vector similarity search
    using Milvus, and finally invokes the `GraphCypherQAChain` with the user query.

    Args:
        max_retries: number of times to retry the query in case of failure (default is 3).
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
        model="o1-mini-2024-09-12",
        temperature=1,
        timeout=None,
        max_retries=2,
    )

    # Refresh schema to ensure it's up-to-date
    graph.refresh_schema()

    # Get Cypher prompt template
    chat_prompt = get_cypher_prompt_template()
    start_time = time.time()
    few_shot_examples = get_similar_paths_from_milvus(graph=graph, user_query=user_query, top_k=5)
    end_time = time.time()
    print(f"****Time taken to conduct vector similarity search in vector DB: {end_time - start_time:.2f} seconds")

    # Create a partial prompt with schema and dataguide_paths filled in. user_query will be filled in later from user query.
    partial_prompt = chat_prompt.partial(
        schema=graph.schema,
        example_queries=few_shot_examples,
        dataguide_paths=formatted_paths
    )
    # retry logic
    retry_count = 0
    queries_and_errors = []
    enhanced_query = user_query

    while retry_count <= max_retries:
        try:
            # Create a fresh chain for each attempt
            chain = GraphCypherQAChain.from_llm(
                cypher_prompt=partial_prompt,
                llm=llm,
                graph=graph,
                verbose=True,
                validate_query=True,
                include_run_info=True,
                return_intermediate_steps=True,
                allow_dangerous_requests=True,  # only use this in development NOT IN PRODUCTION
            )

            # If errors occurred on previous attempts, append error history to form an enhanced query.
            if queries_and_errors:
                error_history = "\n".join(
                    [f"Tried: {q}\nError: {e}" for q, e in queries_and_errors]
                )
                enhanced_query = (
                    f"{user_query}\n\nPreviously I tried these queries with these errors:\n"
                    f"{error_history}\n\nDon't make the same mistakes."
                )

            # Invoke the chain with the enhanced query.
            print("\n****************\nEnhanced Query:\n", enhanced_query)
            response = chain.invoke(enhanced_query)

            # Extract intermediate steps and generated cypher query when present.
            intermediate_steps = response.get("intermediate_steps", [])
            context_data = []
            generated_cypher = None
            # retrieve the context and generated cypher query from the intermediate steps.
            for step in intermediate_steps:
                if "context" in step:
                    context_data = step["context"]
                    break
            if intermediate_steps and not generated_cypher:
                if "query" in intermediate_steps[0]:
                    generated_cypher = intermediate_steps[0]["query"]
                    # Remove leading "cypher\n" if present.
                    if generated_cypher.startswith("cypher\n"):
                        generated_cypher = generated_cypher.replace("cypher\n", "")

            # Retry if context is empty.
            if not context_data:
                # add the generated cypher query and error message to the queries_and_errors list.
                error_msg = f"Empty context returned. Generated cypher: {generated_cypher if generated_cypher else 'No query generated'}"
                queries_and_errors.append((enhanced_query, error_msg))
                retry_count += 1

                if retry_count > max_retries:
                    raise Exception(f"Failed after {max_retries} attempts. Last error: {error_msg}")

                # Update enhanced query with the generated cypher query.
                enhanced_query = (
                    f"{user_query}\n\nPreviously I tried this generated cypher query: {generated_cypher} but it gave me no results. Don't make the same mistake. Take a careful look at dataguide and schema again to ensure you aren't making up paths and following the right sequence\n"
                )
                time.sleep(1)
                continue

            return response

        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            queries_and_errors.append((enhanced_query, error_msg))
            print(
                f"Query failed (attempt {retry_count}/{max_retries})"
            )

            if retry_count > max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {error_msg}")

            time.sleep(1)
