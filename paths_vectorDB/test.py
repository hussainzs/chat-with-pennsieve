import random
from app.database_setup import setup_neo4j_graph
import json

# Connect to Neo4j
graph = setup_neo4j_graph()

def generate_random_paths(graph, num_paths=3):
    # Query to get all nodes except those with label DataGuide (get only the IDs)
    query = "MATCH (n) WHERE NOT n:DataGuide RETURN elementId(n) as id"
    nodes = graph.query(query)

    random_paths = []

    for _ in range(num_paths):
        # Randomly select a node to start
        start_id = random.choice(nodes)['id']
        # print(start_id)

        # Traverse up to the root node
        path_query = f"""
        MATCH path = (p:Pennsieve)-[*]->(n)
        WHERE elementId(n) = "{start_id}"
        RETURN path
        LIMIT 1
        """
        # print(path_query)
        path_result = graph.query(path_query)
        print(json.dumps(path_result, indent=4))
        # random_paths.append(path_result)

    # return random_paths


# Generate and print random paths
generate_random_paths(graph)
