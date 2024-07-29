import random
from app.database_setup import setup_neo4j_graph

# Connect to Neo4j
graph = setup_neo4j_graph()

def generate_random_paths(graph, num_paths=10):
    # Query to get all nodes except those with label DataGuide
    query = "MATCH (n) WHERE NOT n:DataGuide RETURN n"
    nodes = graph.query(query)

    random_paths = []

    for _ in range(num_paths):
        # Randomly select a node to start
        start_node = random.choice(nodes)["n"]
        start_id = start_node.id

        # Traverse up to the root node
        path_query = f"""
        MATCH path = (n)-[:*]->(p:Pennsieve)
        WHERE ID(n) = {start_id}
        RETURN path
        LIMIT 1
        """
        path_result = graph.query(path_query)
        if path_result:
            # Format the path for debugging purposes
            path = path_result[0]["path"]
            formatted_path = "(:Pennsieve)" + "".join(
                f"-[:{rel.type}]->()" for rel in path.relationships
            )
            random_paths.append(formatted_path)
            print(f"Path {len(random_paths)}: {formatted_path}")

    return random_paths


# Generate and print random paths
random_paths = generate_random_paths(graph)
