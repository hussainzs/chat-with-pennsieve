from typing import List
from paths_vectorDB.generate_descriptions import generate_embedding
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import subprocess


def create_and_fill_milvus_collection(collection_name: str, all_paths: List[str], all_descriptions: List[str]):
    """
    Creates and fills the Milvus collection with provided paths and descriptions.

    This function connects to a Milvus instance, creates a collection if it does not exist, and inserts the provided Cypher paths and descriptions into the collection.

    Pitfalls:
        - Ensure that the Milvus container is running before executing this function.
        - Verify that the collection schema matches the data to be inserted.

    Args:
        collection_name (str): The name of the collection to create and fill.
        all_paths (List[str]): List of Cypher paths to be inserted.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.

    Returns:
        None
    """
    connection_alias = "default"
    # step 1: Connect to the Milvus instance
    connections.connect(alias=connection_alias, host='localhost', port='19530')
    # step 3: Create the collection (if it doesn't exist)
    create_collection(collection_name, define_schema(), connection_alias)
    # step 4: Insert data into the collection
    insert_bulk_data(collection_name, all_paths, all_descriptions)
    print("Setup complete✔️✔️✔️ \n")


def start_milvus_using_docker_compose():
    """
    Starts the Milvus container using Docker Compose if it is not already running.

    This function checks if the Milvus container is running. If it is not running, it starts the container using the `docker-compose up -d` command.

    Pitfalls:
        - Ensure that Docker is installed on the host machine and accessible from the terminal.
        - Ensure that the docker-compose.yml file is in the root directory of the project.
    Raises:
        subprocess.CalledProcessError: If there is an error starting the Milvus container.
    """
    try:
        if not is_milvus_container_running():
            # step 1: Start up the Milvus instance using docker compose
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("Milvus container started successfully ✔✔")
    except subprocess.CalledProcessError as e:
        print(f"❌❌❌Error starting Milvus container: {e}")


def collection_exists(collection_name: str) -> bool:
    """
    Checks if a specified Milvus collection exists in the `default` connection.

    This function connects to a Milvus instance using the default connection alias and checks if the specified collection exists.

    Args:
        collection_name (str): The name of the collection to check.

    Returns:
        bool: True if the collection exists, False otherwise.

    Pitfalls:
        - Ensure that the Milvus container is running before executing this function.
        - Handle exceptions properly to avoid unexpected crashes.
    """
    connections.connect(alias="default", host='localhost', port='19530')
    return utility.has_collection(collection_name)


def is_milvus_container_running() -> bool:
    """
    Checks if the Milvus container is running.

    This function runs the `docker ps` command to check if `milvus-standalone` is in the output. If it is, the Milvus container is running.

    Pitfalls:
        - Ensure the docker-compose.yml file is in the root directory of the project.

    Returns:
        bool: True if the Milvus container is running, False otherwise.

    Raises:
        subprocess.CalledProcessError: If there is an error executing the `docker ps` command.
    """
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
    Defines the schema for a Milvus collection.

    This function creates a schema for a Milvus collection with fields for
    1) ID
    2) Cypher path
    3) description
    4) embedding.
    The ID field is the primary key and is auto-generated. The Cypher path and description fields are variable-length strings, and the embedding field is a fixed-dimension float vector.

    Pitfalls:
        - Ensure that the field names and data types match the data to be inserted.
        - Verify that the embedding dimension matches the expected size for the vectors.

    Note: If you change the embedding dimension, you must regenerate the embeddings for the data. Also make sure the `embedding` field matches the embedding model's output dimension

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


def create_collection(collection_name: str) -> Collection | None:
    """
    Creates a Milvus collection IF IT DOES NOT EXIST ALREADY.

    This function connects to a Milvus instance using the specified connection alias, checks if the collection exists, and creates it if it does not. The collection is created with the provided schema.

    Pitfalls:
        - Ensure that the Milvus container is running before executing this function.
        - Ensure a connection with Milvus has been established before calling this function.

    Args:
        collection_name (str): The name of the collection to create.
        schema (CollectionSchema): The Milvus schema to define the structure of the collection.
        connection_alias (str, optional): The alias of the connection to Milvus. Defaults to "default".

    Returns:
        Collection: The created Milvus collection object.

    Raises:
        Exception: If there is an error connecting to the Milvus instance or creating the collection.
    """
    if not utility.has_collection(collection_name):
        connection_alias = "default"
        # step 1: Connect to the Milvus instance
        connections.connect(alias=connection_alias, host='localhost', port='19530')
        # step 3: Create the collection (if it doesn't exist)
        collection = Collection(name=collection_name, schema=define_schema(), using=connection_alias)
        print(f"✔️✔️Collection {collection_name} created.")
        return collection
    else:
        print(f"Collection {collection_name} already exists.")


def insert_bulk_data(collection_name: str, all_paths: List[str], all_descriptions: List[str]) -> bool:
    """
    Inserts Cypher paths, descriptions, and their embeddings into an existing Milvus collection.

    This function connects to an existing Milvus collection, generates embeddings for the provided descriptions using the OpenAI API, and inserts the Cypher paths, descriptions, and embeddings into the collection. It also handles exceptions and ensures data is flushed to disk.

    Pitfalls:
        - Ensure that the Milvus container is running before executing this function.
        - Verify that the collection exists and has the correct schema.

    Args:
        collection_name (str): The name of the Milvus collection.
        all_paths (List[str]): List of Cypher paths to be inserted.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.

    Returns:
        bool: True if the data is successfully inserted, False otherwise.

    Raises:
        Exception: If embedding generation fails for any path.
        Exception: If there is an error during data insertion into Milvus Collection.
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


def insert_single_data(collection_name: str, path: str, description: str) -> bool:
    try:
        existing_collection = Collection(collection_name)
    except Exception as e:
        print(f"Error accessing collection {collection_name}: {e}")
        return False
    vector_embedding = generate_embedding(description)
    if not vector_embedding:
        print(f"Skipping insertion for path due to embedding failure: \n{path}\n")
        return False
    record = {
        "cypher_path": path,
        "description": description,
        "embedding": vector_embedding
    }
    try:
        existing_collection.insert([record])
        existing_collection.flush()
        return True
    except Exception as e:
        print(f"Failed insertion for path: \n{path}\nThe Error was == {e}")
        return False


def remove_collection(collection_name: str) -> None:
    """
    Removes a specified Milvus collection if it exists.

    This function connects to a Milvus instance, checks if the specified collection exists,
    and removes it if it does.

    Note: Connection used has `default` alias and is on localhost:19530. Change it here if necessary.

    Pitfalls:
        -  Ensure that the Milvus container is running before executing this function.
    Args:
        collection_name (str): The name of the collection to remove.

    Returns:
        None

    Raises:
        Exception: Exceptions from the Milvus utility functions.
    """
    connections.connect(alias="default", host='localhost', port='19530')
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name, using="default")
        print(f"Collection {collection_name} dropped.")
    else:
        print(f"Collection {collection_name} does not exist.")


def get_collection_size(collection_name: str) -> int:
    """
    Retrieves the number of entities in a specified Milvus collection.

    This function connects to a Milvus instance and returns the number of entities in the specified collection.

    Args:
        collection_name (str): The name of the collection to check.

    Returns:
        int: The number of entities in the collection.

    Raises:
        Exception: If there is an error connecting to the Milvus instance or accessing the collection.
    """
    connections.connect(alias="default", host='localhost', port='19530')
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        return collection.num_entities
    else:
        raise Exception(f"Collection {collection_name} does not exist.")


def search_similar_vectors(collection_name: str, user_query: str, top_k: int = 3) -> List[str]:
    """
    Conducts a vector similarity search in a specified Milvus collection using the embedding field.

    This function generates an embedding for the user query, connects to the Milvus collection,
    checks for an existing index, and performs a similarity search to find the most similar vectors.
    It prints the distances of the returned hits and returns the Cypher paths of the similar vectors.

    Note: It loads the collection into memory for search and releases it after the search is complete.

    Pitfalls:
        - Ensure that the Milvus container is running before executing this function.
        - Verify that the collection that you are searching in exists.
        - Your API key for OpenAI should be set in the environment variable to enable embedding generation.

    Args:
        collection_name (str): The name of the Milvus collection to search in.
        user_query (str): The user query string to generate the embedding.
        top_k (int, optional): The number of similar vectors to return (in descending order of similarity). Defaults to 3.

    Returns:
        List[str]: A list of Cypher paths that are most similar to the user query.

    Raises:
        Exception: If the embedding generation for the user query fails.
        Exception: If the specified collection does not exist.
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
    if not utility.has_collection(collection_name):
        raise Exception(f"Collection {collection_name} does not exist.")
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
        output_fields=['cypher_path', 'description'],  # Specify the fields to return
    )

    # Step 5: Print distances of the returned hits and store the Cypher paths
    print(f"Success ✔️✔️: Similar vectors are following:")
    if results:
        for i, hit in enumerate(results[0]):
            path: str = hit.entity.get('cypher_path')
            description: str = hit.entity.get('description')

            # Combine them into a formatted string
            formatted = f"cypher query: {path}\ndescription: {description}\n"
            output.append(formatted)

            # Optional debug print
            print(f"Hit {i + 1}:")
            print(f"  Cypher Path: {path}")
            print(f"  Description: {description}")
            print(f"  Distance: {hit.distance}")

    # Step 6: Release the collection to reduce memory consumption
    collection.release()
    print("Collection released from memory.")

    return output
