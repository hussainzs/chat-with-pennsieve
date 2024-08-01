from typing import List, Dict, Any
from generate_descriptions import generate_embedding
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import subprocess
import os
import re
from pymilvus import MilvusClient


def setup_and_create_milvus_collection(collection_name: str, all_paths: List[str], all_descriptions: List[str]):
    connection_alias = "default"
    if not is_milvus_container_running():
        # step 1: Start up the Milvus instance using docker compose
        subprocess.run(["docker-compose", "up", "-d"], check=True)
    # step 2: Connect to the Milvus instance
    connections.connect(alias=connection_alias, host='localhost', port='19530')
    # step 3: Create the collection
    create_collection(collection_name, define_schema(), connection_alias)
    # step 4: Insert data into the collection
    insert_data(collection_name, all_paths, all_descriptions)
    print("Milvus collection created and collection created ✔️✔️✔️")


def connect_to_milvus_without_error_checking():
    if is_milvus_container_running():
        return
    else:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("Milvus container started successfully.")
        # subprocess.run(["docker", "port", "milvus-standalone", "19530/tcp"], check=True)
        milvus_connection = connections.connect(
            alias="default",
            host='localhost',
            port='19530'
        )
        print("✔️✔️✔️✔️✔️Connected to Milvus instance on port 19530")
        return milvus_connection


# I understand this function is written badly. ideally it should be refactored. It kinda works.
def connect_to_milvus_with_error_checking():
    """
    Connect to the Milvus instance. If the instance is not running, start it using Docker.
    use connect_to_milvus_simple() instead for now.

    Args:
        host (str): Host address of the Milvus instance.
        port (str): Port number of the Milvus instance.
    """
    # Check if docker-compose.yml exists in the project root directory (assume: *one level above the current directory*)
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    docker_compose_path = os.path.join(project_root_path, "docker-compose.yml")
    print(f"Docker compose file path: {docker_compose_path}")
    if os.path.exists(docker_compose_path):
        print("✔ docker-compose.yml file found in the project root directory.")
    else:
        print("❌ To run Milvus, please place the docker-compose.yml file in the root directory from "
              "https://github.com/milvus-io/milvus/releases/download/v2.4.6/milvus-standalone-docker-compose.yml")
        return

    # Step 1: Start the Milvus standalone instance using Docker
    try:
        print("Step 1: ...Starting Milvus standalone instance using Docker...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✔ Step 1: Milvus standalone instance started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Step 1: Failed to start Milvus standalone instance: {e}")
        return

    # Step 2: Check for the Milvus standalone process and its default port
    try:
        # Step 2: Check for the Milvus standalone process and its default port
        print("Step 2: ...Running docker compose ps to check for Milvus standalone process...")
        result = subprocess.run(["docker", "compose", "ps"], capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")

        milvus_line = None
        for line in output_lines:
            if "milvus-standalone" in line:
                milvus_line = line
                break

        if milvus_line:
            # Use regex to find the mapping for port 19530
            port_mapping = re.search(r'(\d+)->19530/tcp', milvus_line)
            if port_mapping:
                print(f"✔✔ Step 2: Milvus standalone container found on port 19530")
                port = 19530
            else:
                print(
                    "❌❌ Step 2: We couldn't find the default port 19530. Please run 'docker compose ps' to find the relevant port "
                    "and then run 'docker port milvus-standalone {port_number}/tcp' in your terminal to establish a "
                    "connection with Milvus.")
                port = None
                return
        else:
            print("❌❌ Step 2: 'milvus-standalone' container not found in the output of 'docker compose ps'")
            port = None
            return

    except subprocess.CalledProcessError as e:
        print(f"❌❌ Step 2: Failed to check for Milvus standalone process: {e}")
        port = None
        return

    # Step 3: Connect to the Milvus instance using the obtained port
    try:
        print(f"Step 3: ...Connecting to Milvus instance on port {port}")
        subprocess.run(["docker", "port", "milvus-standalone", "19530/tcp"], check=True)
        print("✔✔✔ Step 3: Connected to Milvus instance.")
    except Exception as e:
        print(f"❌❌❌ Step 3: Failed to connect to Milvus instance: {e}")
        return


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

# def check_connection():
#     """
#     Check the connection status to the Milvus instance.
#
#     Returns:
#         bool: True if connected, False otherwise.
#     """
#     if connections.has_connection("default"):
#         print("***************Milvus is running on port 19530***************")
#         return True
#     else:
#         print("---------------Failed to connect to Milvus---------------")
#         return False


def define_schema():
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


def insert_data(collection_name: str, all_paths: List[str], all_descriptions: List[str]):
    """
    Insert the Cypher paths, descriptions, and embeddings into the Milvus collection.

    Args:
        collection_name (str): The name of the collection.
        all_paths (List[str]): List of Cypher paths.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.
    """
    existing_collection = Collection(collection_name)
    data = []
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
        print(f"Insert result: {insert_result}")
    except Exception as e:
        print(f"Error during insert: {e}")
        return
    # Flush the collection to ensure data is written to disk
    existing_collection.flush()
    print("Data flushed to disk.")

    # Check the number of entities in the collection
    num_entities = existing_collection.num_entities
    print(f"Number of entities in collection after insert: {num_entities}")


def remove_collection(collection_name: str):
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


def disconnect_milvus():
    """
    Disconnect from the Milvus instance.
    """
    connections.disconnect("default")
    print("Disconnected from Milvus instance")