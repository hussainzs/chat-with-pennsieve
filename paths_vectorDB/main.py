from app.database_setup import setup_neo4j_graph
from random_path_generator import generate_formatted_random_paths
from generate_descriptions import generate_path_descriptions
from vectorDB_setup import setup_and_create_milvus_collection
from write_read_data import write_paths_and_descriptions_to_file
from typing import List
import time


#### manually run remove collection from test.pu before running this function if you want to rebuild the collection
def fill_collection_with_random_paths():
    # Create a Neo4jGraph object
    graph = setup_neo4j_graph()

    total_time_taken = 0

    # get random cypher paths and their descriptions in an array of strings
    start_time = time.time()
    print("Generating random paths and formatting them...")
    number_of_paths = 10
    all_paths: List[str] = generate_formatted_random_paths(graph, number_of_paths)
    end_time = time.time()
    total_time_taken += end_time - start_time
    print(
        f"$$$$$$$$$Time taken to generate {number_of_paths} random paths and format them: {total_time_taken:.2f} seconds")
    print("")

    print("Generating descriptions for random paths using OpenAI API...")
    start_time = time.time()
    all_descriptions: List[str] = generate_path_descriptions(all_paths)
    end_time = time.time()
    total_time_taken += end_time - start_time
    print(f"$$$$$$$$$Time taken to generate descriptions for random paths: {total_time_taken:.2f} seconds")
    print("")

    # write the paths and descriptions to a file to save API calls
    write_paths_and_descriptions_to_file(all_paths, all_descriptions)

    # setup milvus connection and create collection for storing paths and descriptions
    start_time = time.time()
    setup_and_create_milvus_collection("default", all_paths=all_paths, all_descriptions=all_descriptions)
    end_time = time.time()
    total_time_taken += f"{total_time_taken:.2f}"
    print(f"$$$$$$$$$Time taken to setup and create Milvus collection: {total_time_taken:.2f} seconds")
    print("")
    print(f"Total time taken to\n"
          f"1. generate {number_of_paths} random paths + \n"
          f"2. format the random paths into cypher queries + \n"
          f"3. generate their descriptions using LLM + \n"
          f"4. Connect to Milvus, create embeddings and store the data in Milvus"
          f" = : {{{total_time_taken:.2f}}} seconds")

