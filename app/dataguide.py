from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph


def extract_dataguide_paths(graph: Neo4jGraph) -> List[Dict[str, Any]]:
    query = """
    MATCH path = (root:DataGuide:Root)-[*]->(leaf:DataGuide)
    WHERE NOT (leaf)-->()
    RETURN path
    """
    results = graph.query(query)
    return results


def format_paths_for_llm(results: List[Dict[str, Any]]) -> List[str]:
    formatted_paths = []
    for record in results:
        path = record['path']
        path_elements = ["(:Pennsieve)"]  # prepend root element
        for rel in path:
            if isinstance(rel, str):  # ignores the nodes represented by {} in the path
                if rel.isdigit():
                    path_elements.append(f"-[:`{rel}`]->()")
                else:
                    path_elements.append(f"-[:{rel}]->()")
        formatted_paths.append("".join(path_elements))
    return formatted_paths
