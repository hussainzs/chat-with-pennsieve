from app.database_setup import setup_neo4j_graph
from random_path_generator import generate_formatted_random_paths
from generate_descriptions import generate_path_descriptions
from vectorDB_setup import setup_milvus_collection, insert_data, disconnect_milvus
from typing import List

# Create a Neo4jGraph object
graph = setup_neo4j_graph()

# get random cypher paths and their descriptions in an array of strings
all_paths: List[str] = generate_formatted_random_paths(graph, 2)
all_descriptions: List[str] = generate_path_descriptions(all_paths)

# setup milvus connection and create collection for storing paths and descriptions
collection_name = "VectorDB for Cypher paths and descriptions"
setup_milvus_collection(collection_name)
insert_data(collection_name, all_paths, all_descriptions)
