from langchain_community.graphs import Neo4jGraph
from paths_vectorDB.random_path_generator import generate_formatted_random_paths
from paths_vectorDB.generate_descriptions import generate_path_descriptions
from paths_vectorDB.vectorDB_setup import (start_milvus_using_docker_compose,
                                           search_similar_vectors, remove_collection,
                                           collection_exists, insert_single_data, create_collection,
                                           get_collection_size)
from paths_vectorDB.write_read_data import write_paths_and_descriptions_to_file
from typing import List
import time
from pymilvus import connections, Collection


# assumes Milvus instance is running
def fill_collection_with_random_paths(graph: Neo4jGraph, collection_name: str, num_of_paths: int,
                                      rebuild_collection: bool = False) -> None:
    """
    Fills the Milvus collection with random paths and their descriptions. If rebuild_collection is True, the current
    collection will be deleted and new collection will be created and filled with random paths and descriptions.
    Note: setting rebuilt_collection = True may input duplicate data into Milvus or misbehave. It wasn't tested during development.

    Pitfalls:
        -  Make sure you run start_milvus_using_docker_compose() before calling this function. Or in terminal run `docker-compose up -d` to start Milvus (if it's not currently running).

    Args:
        graph (Neo4jGraph): The Neo4jGraph object used to generate random paths from.
        collection_name (str): Name of the collection that needs to be created, filled or built.
        num_of_paths (int, optional): Number of randomly generated paths to be used to fill the collection.
        rebuild_collection (bool, optional): Fill New collection if True, else append to the existing collection. Defaults to False

    Returns: None

    """

    total_time_taken = 0

    # Step 1: Generate random paths
    print("Step 1: Generating random paths...")
    start_time = time.time()
    all_paths = generate_formatted_random_paths(graph, num_of_paths)
    gen_time = time.time() - start_time
    total_time_taken += gen_time
    print(f"Generated {num_of_paths} random paths from Neo4j in {gen_time:.2f} seconds.")

    # Step 2: Optionally rebuild the collection
    if rebuild_collection:
        print("Step 2: Removing current collection (if any).")
        try:
            start_time = time.time()
            remove_collection(collection_name)
            rm_time = time.time() - start_time
            total_time_taken += rm_time
            print(f"Collection {collection_name} removed in {rm_time:.2f} seconds.")
        except Exception as e:
            print(f"ERROR: Failed to remove collection {collection_name}: {e}")

    # Step 3: Process each path individually
    for idx, path in enumerate(all_paths, start=1):

        # stop for 5 seconds every 10 paths to avoid overwhelming the system
        if idx % 10 == 0:
            print(f"\n\n\nPausing for 5 seconds to avoid overwhelming API...")
            time.sleep(5)

        print(f"\nProcessing path {idx}:")

        # 3a: Generate description for the path
        print(f"  Generating description for path {idx}. Calling API ...")
        try:
            start_time = time.time()
            description_list = generate_path_descriptions([path])
            description = description_list[0]
            desc_time = time.time() - start_time
            total_time_taken += desc_time
            print(f"  Description for path #{idx} generated in {desc_time:.2f} seconds.")
        except Exception as gen_err:
            print(f"  ERROR: First attempt failed for path {idx}: {gen_err}. Retrying after 3 seconds...")
            time.sleep(3)  # wait for 3 seconds before retrying
            # Retry generating the description one more time
            try:
                start_time = time.time()
                description_list = generate_path_descriptions([path])
                description = description_list[0]
                desc_time = time.time() - start_time
                total_time_taken += desc_time
                print(f"  Description for path #{idx} generated in {desc_time:.2f} seconds on retry.")
            except Exception as retry_err:
                print(f"  ERROR: Skipping path {idx} after retry failure: {retry_err}")
                continue

        # 3b: Insert path and description into Milvus
        print(f"  Inserting path {idx} into the vector DB.")
        start_time = time.time()
        if insert_single_data(collection_name, path, description):
            ins_time = time.time() - start_time
            total_time_taken += ins_time
            print(f"  Path {idx} filled in DB with its description DB in {ins_time:.2f} seconds.")

        # 3c: Write path and description to the file
        print(f"  Writing path {idx} info to the file.")
        try:
            start_time = time.time()
            write_paths_and_descriptions_to_file([path], [description])
            file_time = time.time() - start_time
            total_time_taken += file_time
            print(f"  Path written to file in {file_time:.2f} seconds.")
        except Exception as file_err:
            print(f"  ERROR: Skipping file write for path {idx} due to error: {file_err}")

    print(f"\nTotal time taken for processing: {total_time_taken:.2f} seconds.")

    # print the state of the collection after all the insertions
    try:
        print("Final state of the collection after all the insertions:")
        # Connect to the Milvus server
        connections.connect(host="localhost", port="19530")

        # Access the collection named `default`
        collection = Collection("default")

        # Get the number of elements in the collection
        num_elements = collection.num_entities
        print("Number of elements in the collection:", num_elements)
    except Exception as e:
        print(f"ERROR: Failed to access the collection to check the final state: {e}")


def get_similar_paths_from_milvus(graph: Neo4jGraph, user_query: str, collection_name: str = "default", top_k: int = 5,
                                  number_of_paths: int = 45, rebuild_collection: bool = False) -> List[str]:
    """
    Wrapper function to get similar paths from a Milvus collection. This function is called from `app/qa_chain.py`.
    It starts Milvus if it is not running, rebuilds the collection if it does not exist, or checks the collection size
    and fills it with additional random paths if necessary before conducting a similarity search.

    Args:
        graph (Neo4jGraph): The Neo4jGraph object used to generate random paths from.
        user_query (str): The user query to search for similar paths.
        collection_name (str, optional): The name of the Milvus collection to search in. Defaults to "default".
        top_k (int, optional): The number of similar vectors to return (in descending order of similarity). Defaults to 5.
        number_of_paths (int, optional): The desired number of randomly generated paths in the collection.
        rebuild_collection (bool, optional): If True, the current collection will be deleted and a new collection will be created
                                             and filled with random paths and descriptions. Defaults to False.

    Returns:
        List[str]: A list of similar paths from the Milvus collection.
    """
    # Start Milvus if not running
    start_milvus_using_docker_compose()

    # Check if collection needs rebuilding or creation
    if rebuild_collection or not collection_exists(collection_name):
        print(f"A Collection does not exist or needs rebuild. Creating collection '{collection_name}' now...")
        # Create a new collection and fill it with random paths and descriptions
        create_collection(collection_name)
        fill_collection_with_random_paths(graph=graph, collection_name=collection_name, num_of_paths=number_of_paths,
                                          rebuild_collection=rebuild_collection)
    else:
        # Collection exists. Check how many paths are already present
        try:
            current_size = get_collection_size(collection_name)
            print(f"Collection '{collection_name}' exists with {current_size} entries.")
            if current_size < number_of_paths:
                difference_of_paths = number_of_paths - current_size
                print(f"Collection needs additional {difference_of_paths} paths to meet the defined number of paths. Filling collection now...")
                fill_collection_with_random_paths(graph=graph, collection_name=collection_name, num_of_paths=difference_of_paths,
                                                  rebuild_collection=rebuild_collection)
            else:
                print(f"Collection '{collection_name}' already has {current_size} entries, which is sufficient.")
        except Exception as e:
            print(f"Failed to get collection size for '{collection_name}': {e}")
            # Optionally, you may want to rebuild or handle this case differently.

    # Search for similar paths in the Milvus collection
    return search_similar_vectors(collection_name, user_query, top_k)
