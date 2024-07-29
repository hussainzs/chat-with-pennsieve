from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

# Connect to the Milvus instance
connections.connect("default", host="0.0.0.0", port="19530")

# Check connection otherwise print error message
if connections.has_connection("default"):
    print("***************Milvus is running on port 19530***************")
else:
    print("---------------Failed to connect to Milvus---------------")

# Define the schema for the Milvus collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="cypher_path", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]

schema = CollectionSchema(fields=fields, description="VectorDB for Cypher paths and descriptions")

# Create the collection else print already exists
collection_name = "cypher_paths"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection {collection_name} created.")
else:
    print(f"Collection {collection_name} already exists.")

def disconnect_milvus():
    connections.disconnect("default")
    print("Disconnected from Milvus instance")


