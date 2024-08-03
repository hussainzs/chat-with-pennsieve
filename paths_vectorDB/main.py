from langchain_community.graphs import Neo4jGraph
from paths_vectorDB.random_path_generator import generate_formatted_random_paths
from paths_vectorDB.generate_descriptions import generate_path_descriptions
from paths_vectorDB.vectorDB_setup import (create_and_fill_milvus_collection, start_milvus_using_docker_compose,
                                           search_similar_vectors, remove_collection,
                                           collection_exists)
from paths_vectorDB.write_read_data import write_paths_and_descriptions_to_file
from typing import List
import time


# assumes Milvus instance is running
def fill_collection_with_random_paths(graph: Neo4jGraph, collection_name: str, num_of_paths: int,
                                      rebuild_collection: bool = False):
    total_time_taken = 0

    # get random cypher paths and their descriptions in an array of strings
    start_time = time.time()
    print("Generating random paths and formatting them...")
    all_paths: List[str] = generate_formatted_random_paths(graph, num_of_paths)
    end_time = time.time()
    total_time_taken += end_time - start_time
    print(
        f"$$$$$$$$$Time taken to generate {num_of_paths} random paths and format them: {total_time_taken:.2f} seconds")
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

    # check if rebuild_collection is True, if so, delete the collection and recreate it
    if rebuild_collection:
        print("Removing current collection if it exists...")
        start_time = time.time()
        remove_collection("default")
        end_time = time.time()
        total_time_taken += end_time - start_time
        print(f"$$$$$$$$$Time taken to remove current collection: {total_time_taken:.2f} seconds")
        print("")

    print("Setting up Milvus connection...")
    start_time = time.time()
    create_and_fill_milvus_collection(collection_name, all_paths=all_paths, all_descriptions=all_descriptions)
    end_time = time.time()
    total_time_taken += end_time - start_time
    print(f"$$$$$$$$$Time taken to create and fill Milvus collection: {total_time_taken:.2f} seconds")
    print("")
    print(f"Total time taken to\n"
          f"1. generate {num_of_paths} random paths + \n"
          f"2. format the random paths into cypher queries + \n"
          f"3. generate their descriptions using LLM + \n"
          f"4. Connect to Milvus, create embeddings and store the data in Milvus"
          f" = : {{{total_time_taken:.2f}}} seconds")


def get_similar_paths_from_milvus(graph: Neo4jGraph, user_query: str, collection_name: str = "default", top_k: int = 3,
                                  number_of_paths: int = 10, rebuild_collection: bool = False):
    # If milvus contained isn't running then start it
    start_milvus_using_docker_compose()

    if not collection_exists(collection_name):
        print(f"Collection does not exist. Creating collection '{collection_name}' and filling Milvus with random paths and descriptions...")
    if rebuild_collection or not collection_exists(collection_name):
        fill_collection_with_random_paths(graph=graph, collection_name=collection_name, num_of_paths=number_of_paths,
                                          rebuild_collection=rebuild_collection)
    # search for similar paths in the Milvus collection
    search_similar_vectors(collection_name, user_query, top_k)
