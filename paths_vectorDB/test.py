from typing import List
import time
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import utility, connections
from random_path_generator import format_path_into_cypher, generate_formatted_random_paths
from app.database_setup import setup_neo4j_graph
from generate_descriptions import generate_path_descriptions
from vectorDB_setup import setup_and_create_milvus_collection, insert_data, define_schema, create_collection, remove_collection, search_similar_vectors


# # remove collection default
# remove_collection("default")

# Test vector similarity search
def test_vector_similarity_search():
    search_similar_vectors("What are the last names of contributors in dataset named Test Dataset CNT?")
test_vector_similarity_search()

# # Test paths_vectorDB setup
# def dummy_path_description_generation() -> List[List[str]]:
#     dummyPath1 = "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})"
#     dummyPath2 = "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})"
#     dummyDescription1 = "something 1"
#     dummyDescription2 = "something 2"
#     return [[dummyPath1, dummyPath2], [dummyDescription1, dummyDescription2]]
# def test_vectorDB_insert():
#     dummyData = dummy_path_description_generation()
#     # Create a Milvus collection
#     collection_name = "test"
#     connection_alias = "default"
#     connections.connect(alias=connection_alias, host='localhost', port='19530')
#     if utility.has_collection(collection_name):
#         print("Test Collection already exists, Dropping it now...")
#         utility.drop_collection(collection_name, using="default")
#     else:
#         print("Test Collection does not already exist, Creating one now...")
#     schema = define_schema()
#     collection = create_collection(collection_name, schema, connection_alias)
#     print(f"List of all collections in current Mivlus server: {utility.list_collections(using='default')}")
#     print(f"Test collection schema = {collection.schema}")
#     print(f"Test collection name = {collection.name}")
#     print(f"Test collection description = {collection.description}")
#     print(f"Test collection primary field = {collection.primary_field}")
#     print(f"Is test collection empty? = {collection.is_empty}")
#     print(f"Indexes in test collection = {collection.indexes}")
#     insert_data("test", dummyData[0], dummyData[1])
#     print(f"Is test collection empty after insertion of data? = {collection.is_empty}")
# # test_vectorDB_insert()
#
# def test_setup_and_create_milvus_collection():
#     name = "test"
#     connections.connect(alias="default", host='localhost', port='19530')
#     if utility.has_collection(name):
#         print("Test Collection already exists, Dropping it now...")
#         utility.drop_collection(name, using="default")
#     else:
#         print("Test Collection does not already exist, Creating one now...")
#     dummyData = dummy_path_description_generation()
#     setup_and_create_milvus_collection(name, dummyData[0], dummyData[1])
# # test_setup_and_create_milvus_collection()



# # Test description generation for random paths using OpenAI API
# def test_description_generation(num_of_paths: int):
#     # Load environment variables
#     load_dotenv()
#
#     # Create a Neo4jGraph object
#     graph = setup_neo4j_graph()
#
#     # Generate random paths
#     result_descriptions: List[str] = []
#     all_paths = generate_formatted_random_paths(graph, num_of_paths)
#     result_descriptions = generate_path_descriptions(generate_formatted_random_paths(graph, num_of_paths))
#     for i, description in enumerate(result_descriptions):
#         print(f"Path {i + 1}: {all_paths[i]}")
#         print(f"Description {i + 1}: {description}")
#         print("---------------------------------------")
# test_description_generation(2)


# # Test random_path_generator.py
# def test_random_path_generator(num_of_paths: int):
#     # Create a Neo4jGraph object
#     graph = setup_neo4j_graph()
#     # Generate random paths
#     formatted_paths = generate_formatted_random_paths(graph, num_of_paths)
#     for i, path in enumerate(formatted_paths):
#         print(f"Path {i + 1}: {path}")
# test_random_path_generator(2)


# # test embedding generation for path descriptions
# def test_embedding_generation(sample_description: str):
#     # Load environment variables
#     load_dotenv()
#
#     # Create a Neo4jGraph object
#     graph = setup_neo4j_graph()
#
#     # Test OpenAI API connection
#     api_key = os.environ['OPENAI_API_KEY']
#     if api_key:
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key, dimensions=512)
#         result_vector = embeddings.embed_query(sample_description)
#         print(f"Size: {len(result_vector)}, vector: {result_vector}")
#     else:
#         print("OpenAI API key is not set. Please check your .env file.")
#
#
# # Test embedding generation
# test_embedding_generation("This path queries the first names of contributors in our database.")



# def test_path_from_specific_id(start_id: str):
#     graph = setup_neo4j_graph()
#     path_query = f"""
#             MATCH path = (p:Pennsieve)-[*]->(n)
#             WHERE elementId(n) = "{start_id}"
#             RETURN path
#             LIMIT 1
#             """
#     path_result = graph.query(path_query)
#     # print(path_result)
#     formatted_path = format_path_into_cypher(path_result[0]['path'])
#     print(formatted_path)
#
#
# # Test path generation from a specific node ID
# start_time = time.time()
# test_path_from_specific_id("4:44d5534a-1d00-49f8-b4ef-1576d0080f3e:54")
# end_time = time.time()
# print(f"Function format_cypher() executed in {(end_time - start_time):.2f} seconds")
