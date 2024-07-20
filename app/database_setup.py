from langchain_community.graphs import Neo4jGraph
from app.config import Config


def setup_neo4j_graph():
    return Neo4jGraph(
        url=Config.NEO4J_URI,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD
    )
