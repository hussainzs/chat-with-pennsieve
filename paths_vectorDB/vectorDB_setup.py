from typing import List
from paths_vectorDB.generate_descriptions import generate_embedding
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import subprocess


def create_and_fill_milvus_collection(collection_name: str, all_paths: List[str], all_descriptions: List[str]):
    connection_alias = "default"
    # step 1: Connect to the Milvus instance
    connections.connect(alias=connection_alias, host='localhost', port='19530')
    # step 3: Create the collection (if it doesn't exist)
    create_collection(collection_name, define_schema(), connection_alias)
    # step 4: Insert data into the collection
    insert_data(collection_name, all_paths, all_descriptions)
    print("Setup complete✔️✔️✔️ \n")


def start_milvus_using_docker_compose():
    try:
        if not is_milvus_container_running():
            # step 1: Start up the Milvus instance using docker compose
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("Milvus container started successfully ✔✔")
    except subprocess.CalledProcessError as e:
        print(f"❌❌❌Error starting Milvus container: {e}")


def collection_exists(collection_name: str) -> bool:
    connections.connect(alias="default", host='localhost', port='19530')
    return utility.has_collection(collection_name)


def is_milvus_container_running() -> bool:
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, check=True)
        output = result.stdout
        result = "milvus-standalone" in output
        if result:
            print("Milvus container is already running ✔✔")
            return True
        else:
            print("Milvus container is not currently running --------")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌❌❌Error checking Milvus container status: {e}")
        return False


def define_schema() -> CollectionSchema:
    """
    Define the schema for the Milvus collection.

    Returns:
        CollectionSchema: The schema for the Milvus collection.
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="cypher_path", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    return CollectionSchema(fields=fields, description="VectorDB for Cypher paths and descriptions")


def create_collection(collection_name: str, schema: CollectionSchema, connection_alias: str) -> Collection:
    """
    Create the Milvus collection if it does not exist.

    Args:
        collection_name (str): The name of the collection.
        schema (CollectionSchema): The schema for the collection.
        connection_alias (str): The alias of the connection to Milvus.
    """
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema, using=connection_alias)
        print(f"✔️✔️Collection {collection_name} created.")
        return collection
    else:
        print(f"Collection {collection_name} already exists.")


def insert_data(collection_name: str, all_paths: List[str], all_descriptions: List[str]) -> bool:
    """
    Insert the Cypher paths, descriptions, and embeddings into an existing Milvus collection.
    Creates embeddings of the descriptions using the OpenAI API and inserts them as well.

    Args:
        collection_name (str): The name of the collection.
        all_paths (List[str]): List of Cypher paths.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.
    """
    existing_collection = Collection(collection_name)
    data = []
    print(f"Number of entities in collection before insert: {existing_collection.num_entities}")
    print("Inserting new data into collection...")
    for index in range(len(all_paths)):
        path = all_paths[index]
        description = all_descriptions[index]
        vector_embedding = generate_embedding(description)
        if not vector_embedding:
            print(f"Failed to generate embedding for path {path}")
            raise Exception("Failed to generate embedding, because generate_descriptions.generate_embedding() returned []")
        dictionary = {
            "cypher_path": path,
            "description": description,
            "embedding": vector_embedding
        }
        data.append(dictionary)
    # Insert data into the collection
    try:
        insert_result = existing_collection.insert(data)
        print(f"Insertion Successful ✔️✔️: Insert result: {insert_result}")
    except Exception as e:
        print(f"Insertion Failure ❌❌: Error during insert: {e}")
        return False
    # Flush the collection to ensure data is written to disk immediately
    existing_collection.flush()
    print("Data flushed to disk✔️✔️")

    # Check the number of entities in the collection
    num_entities = existing_collection.num_entities
    print(f"Number of entities in collection after insert: {num_entities}")

    return True


def remove_collection(collection_name: str) -> None:
    """
    Remove the Milvus collection.

    Args:
        collection_name (str): The name of the collection to remove.
    """
    connections.connect(alias="default", host='localhost', port='19530')
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name, using="default")
        print(f"Collection {collection_name} dropped.")
    else:
        print(f"Collection {collection_name} does not exist.")


def search_similar_vectors(collection_name: str, user_query: str, top_k: int = 3) -> List[str]:
    """
    Conduct a vector similarity search with the embedding field in the collection named 'default'.

    Args:
        top_k: the number of similar vectors to return (in descending order of similarity).
        user_query (str): The user query string.
    """
    output: List[str] = []  # List of Cypher paths

    # Step 1: Generate embedding for the user query
    print("Generating embedding for the user query...")
    user_query_vector = generate_embedding(user_query)
    if not user_query_vector:
        print("Failed to generate embedding for the user query.")
        raise Exception("Failed to generate embedding for the user query.")

    # Step 2: Get an existing collection and load it
    connections.connect(alias="default", host='localhost', port='19530')
    collection = Collection(collection_name)
    print(f"Collection established with Milvus collection named '{collection_name}'.")

    # Step 3: Check if the collection has an index
    if not collection.indexes:
        print("Index doesn't exist, creating index...")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Index created on the 'embedding' field in '{collection_name}' collection.")
    else:
        print(f"✔️✔️ Index already exists on the embedding field in '{collection_name}' collection.")

    print("\nLoading the collection into memory for search...")
    collection.load()

    # Step 3: Define search parameters
    search_params = {
        "metric_type": "IP",
    }

    # Step 4: Conduct the search
    print(f"Searching for similar vectors to user query in the '{collection_name}' collection...")
    results = collection.search(
        data=[user_query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=['cypher_path'],
    )

    # Step 5: Print distances of the returned hits and store the Cypher paths
    print(f"Success ✔️✔️: Similar vectors are following:")
    if results:
        for i, hit in enumerate(results[0]):
            path: str = hit.entity.get('cypher_path')
            output.append(path)
            print(f"Hit {i+1}:")
            print(f"  Cypher Path: {path}")
            print(f"  Distance: {hit.distance}")

    # Step 6: Release the collection to reduce memory consumption
    collection.release()
    print("Collection released from memory.")

    return output
