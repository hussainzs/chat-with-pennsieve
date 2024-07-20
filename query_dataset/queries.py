import json

# Define the dataset of user questions and corresponding correct Cypher queries
few_shot_examples = [
    {
        "user_question": "What are the first names and last names of contributors in our database?",
        "cypher_query": """MATCH (p:Pennsieve)-[:DATASET]->()-[:FILES]->()-[:DATA]->()-[:contributors]->()-[:first_name]->(fn:Data),
              (p)-[:DATASET]->()-[:FILES]->()-[:DATA]->()-[:contributors]->()-[:last_name]->(ln:Data)
              RETURN DISTINCT fn.value AS first_name, ln.value AS last_name"""
    },
    {
        "user_question": "List the names of all datasets in the Pennsieve database.",
        "cypher_query": """MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset)
        RETURN DISTINCT d.name AS dataset_name"""
    },
    {
        "user_question": "Does dataset named: Test Dataset CNT has banner.jpg file?.",
        "cypher_query": """MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset {name: "Test Dataset CNT"})-[:FILES]->(f:File {name: "banner.jpg"})
        RETURN d.name, f.name"""
    },
    {
        "user_question": "Get the number of files in each dataset",
        "cypher_query": """MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset)-[:FILES]->(f:File)
        RETURN d.name AS dataset_name, COUNT(f) AS file_count"""
    },
    {
        "user_question": "What is the description of dataset with id 379",
        "cypher_query": """MATCH (p:Pennsieve)-[:DATASET]->(d:Dataset {id: 379})-[:FILES]->()-[:DATA]->()-[:description]->(desc:Data)
        RETURN desc.value AS description"""
    }
]


def preprocess_cypher_query(query):
    lines = query.split('\n')
    stripped_lines = [line.strip() for line in lines]
    return '\n'.join(stripped_lines)


# Preprocess each cypher_query in your dataset
for example in few_shot_examples:
    example['cypher_query'] = preprocess_cypher_query(example['cypher_query'])

# Save the preprocessed dataset to a JSON file
# with open('few_shot_examples.json', 'w') as file:
#     json.dump(few_shot_examples, file, indent=4)
#print("Few-shot examples dataset created and saved to 'few_shot_examples.json'.")

def get_few_shot_examples():
    return few_shot_examples


