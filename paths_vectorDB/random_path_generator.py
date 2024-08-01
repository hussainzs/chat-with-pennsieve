import random
import json
from typing import List, Dict, Union, Any
from langchain_community.graphs import Neo4jGraph


def generate_formatted_random_paths(graph: Neo4jGraph, num_of_paths: int) -> List[str]:
    """
    Generates random paths from the graph database and returns a list of formatted Cypher paths.
    Calls the generate_random_paths() function to get the raw paths and
    formats them using format_path_into_cypher().

    Args:
        graph: Neo4j graph object (from LangChain Neo4j wrapper to enable the .query() method).
        num_of_paths: Number of random paths to generate.

    Returns:
        List[str]: List of formatted Cypher paths as strings.
    """

    raw_paths = generate_random_paths(graph, num_of_paths)
    result: List[str] = []
    for path_data in raw_paths:
        path_elements: List[Union[str, Dict[str, Any]]] = path_data[0]['path']
        formatted_path = format_path_into_cypher(path_elements)
        result.append(formatted_path)
    return result


def generate_random_paths(graph, num_paths: int) -> List[List[Dict[str, Any]]]:
    """
    Generates random paths from the graph database by choosing a random node and going up to the root node.
    Returns paths as a list of List of dictionaries. **Refer to sample_paths variable at the bottom for an example output of 2 random paths**
    Return Format: List of dictionaries, where each dictionary contains a key 'path' that maps to a List of mixed types (strings {relationship names}, and dictionaries {node properties}).
    Link to LangChain Neo4j wrapper documentation: https://api.python.langchain.com/en/latest/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html#langchain_community.graphs.neo4j_graph.Neo4jGraph.query
    Note: It does not add any nodes with the label DataGuide.

    Args:
        graph: Neo4j graph object (from LangChain Neo4j wrapper to enable the .query() method).
        num_paths: Number of random paths to generate.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the paths (for an example, see the sample_paths variable at the bottom).
    """
    # Query to get all nodes except those with label DataGuide (get only the IDs)
    query = "MATCH (n) WHERE NOT n:DataGuide RETURN elementId(n) as id"
    nodes = graph.query(query)

    all_paths = []

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
        path_result = graph.query(path_query)
        # print(path_result)
        # print(json.dumps(path_result, indent=4))
        all_paths.append(path_result)

    return all_paths


"""
*** Keep this comment right above format_path_into_cypher() function ***
sample_input = [
    {},
    "DATASET",
    {
        "name": "Test Dataset CNT",
        "id": 214.0
    },
    "FILES",
    {
        "name": "schema.json"
    },
    "DATA",
    {
        "children": 2.0,
        "type": "Object"
    },
    "models",
    {
        "children": 1.0,
        "type": "Array"
    },
    "INDEX",
    {
        "children": 5.0,
        "type": "Object"
    },
    "file",
    {
        "value": "records/file.csv"
    }
]
sample_output = "(:Pennsieve) - [:DATASET]->(:Dataset {name: 'Test Dataset CNT'}) - [: FILES]->
    (:File {name: 'schema.json'}) - [: DATA]->(:Data {children: 2.0, type: 'Object'}) - [: models]->
    (:Data {children: 1.0, type: 'Array'}) - [: INDEX]->(:Data {children: 5.0, type: 'Object'}) - [: file]->
    (:Data {value: 'records/file.csv'})"
"""


def format_path_into_cypher(path_elements: List[Union[str, Dict[str, Any]]]) -> str:
    """
    Formats a given path into Cypher path syntax.

    This function accepts a list of path elements and returns a formatted Cypher path string.
    For detailed examples of input and output, see the comment above this function.

    Args:
        path_elements (List[Union[str, Dict[str, Any]]]): List containing the path elements of a single path.

    Returns:
        str: Formatted Cypher path as a string.
    """
    path_parts: List[str] = ["(:Pennsieve)"]  # Initialize with the starting label
    for i in range(1, len(path_elements), 2):
        relationship: str = path_elements[i]
        node_properties: Dict[str, Any] = path_elements[i + 1]  # empty {} if no properties

        # Determine the node label based on the relationship type
        if relationship == "DATASET":
            node_label = "Dataset"
        elif relationship == "FILES":
            node_label = "File"
        else:
            node_label = "Data"

        # Extract and format properties for the node
        # i.e. name: 'Test Dataset CNT', children: 2.0, type: 'Object' etc
        props: List[str] = []
        for k, v in node_properties.items():
            if k != "id":  # Skip the 'id' property
                if isinstance(v, (int, float)):
                    props.append(f"{k}: {v}")
                else:
                    props.append(f"{k}: '{v}'")

        # Join the formatted properties into a single string
        # put everything in curly braces if properties exist {name: 'Test Dataset CNT', children: 2.0}
        props_str = ", ".join(props)
        if props_str:
            props_str = f" {{{props_str}}}"  # use double {} to add curly braces in the string

        # Format the relationship, adding backticks if it is numeric i.e. relationship `3`
        if relationship.isdigit():
            relationship = f"`{relationship}`"

        # Append the formatted relationship and node to the list
        path_parts.append(f"-[:{relationship}]->(:{node_label}{props_str})")

    # Join all parts into a single string at the end
    formatted_path: str = "".join(path_parts)
    return formatted_path

"""
sample_paths = [
    [{'path': [{}, 'DATASET', {'name': 'Test Dataset CNT', 'id': 214.0}, 'FILES', {'name': 'test.edf'}, 'DATA', 
    {'children': 3.0, 'type': 'Object'}, '_physicalSignals', {'children': 12.0, 'type': 'Array'}, 'INDEX', 
    {'children': 600.0, 'type': 'Array'}, 'INDEX', {'children': 200.0, 'type': 'Object'}, '9', 
    {'value': 95.59777069091797}]}], 
    [{'path': [{}, 'DATASET', {'name': 'Test Dataset CNT', 'id': 214.0}, 'FILES', {'name': 'test.edf'}, 
    'DATA', {'children': 3.0, 'type': 'Object'}, '_rawSignals', {'children': 12.0, 'type': 'Array'}, 'INDEX', 
    {'children': 600.0, 'type': 'Array'}, 'INDEX', {'children': 200.0, 'type': 'Object'}, '6', {'value': 0.0}]}]]

"""
