from dotenv import load_dotenv
from app.database_setup import setup_neo4j_graph
from app.dataguide import extract_dataguide_paths, format_paths_for_llm
import time
import json

# Load environment variables
load_dotenv()

### Test extracting and formatting dataguide paths
print("Setting up graph...")
start = time.time()
graph = setup_neo4j_graph()
end = time.time()
print(f"Time to setup graph: {(end - start):.4f} seconds\n")

print("\n\nNeo4j Schema:")
start = time.time()
print(graph.schema)
end = time.time()
print(f"Time to print schema: {(end - start):.4f} seconds\n")



# print("Extracting dataguide paths...")
# start = time.time()
# unformatted_paths = extract_dataguide_paths(graph)
# end = time.time()
# print(f"Time to extract dataguide paths: {(end - start):.4f} seconds, Here are the extracted unformatted paths: ")
# print(unformatted_paths)
# print("")
#
# print("Formatting paths for LLM...")
# start = time.time()
# paths = format_paths_for_llm(unformatted_paths)
# end = time.time()
# print(f"Time to format paths for LLM: {(end - start):.4f}  seconds")
# print(f"Here are the formatted paths: \n {json.dumps(paths, indent=2)} \n")
