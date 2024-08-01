from neo4j import GraphDatabase

# Replace with your Neo4j connection details
uri = "bolt://localhost:7687"
username = "neo4j"
password = "123456789"


def run_query(tx, start_id):
    query = """
    MATCH path = (p:Pennsieve)-[*]->(n)
    WHERE elementId(n) = "4:44d5534a-1d00-49f8-b4ef-1576d0080f3e:10"
    RETURN path
    LIMIT 1
    """
    result = tx.run(query, start_id=start_id)
    print("*********************************************************")
    for record in result:
        print(record)
    print("*********************************************************")
    return [record["path"] for record in result]


with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    with driver.session() as session:
        start_id = "4:44d5534a-1d00-49f8-b4ef-1576d0080f3e:10"
        paths = session.read_transaction(run_query, start_id)

        for path in paths:
            print(str(path))