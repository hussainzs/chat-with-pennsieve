from langchain.schema import HumanMessage, SystemMessage
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# System instructions template string
system_message_str = """
You are a Neo4j expert specializing in medical datasets and files.

**Your Task:**  
Given a Cypher path in the graph that starts from the root node and traverses to a specific node, your task is to provide a concise and specific description of the data point or information retrievable from that path.

These descriptions must be clear and domain-specific. Include relevant keywords from Cypher properties, node labels, and relationships (e.g., EDF, header, value, type, index). Avoid filler words. The goal is to educate another LLM with no prior knowledge of the graph structure.

**Graph Structure:**  
1. `:Pennsieve` (root) connects to `:Dataset` via the `:DATASET` relationship.  
2. `:Dataset` connects to `:Directory` or `:File` via the `:FILES` relationship.  
3. `:Dataset`, `:Directory`, and `:File` nodes have a `name` property used for filtering.  
4. `:File` nodes connect to `:Data` via the `:DATA` relationship.  
5. `:Dataset` nodes have an `id` property representing the Pennsieve dataset ID.  
6. All nodes beyond `:Dataset` and `:FILES` relationships are labeled `:Data`.  
7. Only leaf `:Data` nodes contain a `value` property, used for filtering/conditioning.  
8. `:INDEX` relationships include an `index` property used for positional access.  
9. `:Data` node properties:  
   - `children`: Count of child relationships (array elements or object keys).  
   - `type`: Indicates data structure type (`Array` or `Object`).  
   - `value`: Present only in leaf nodes.  

**Graph Semantics:**  
- All key-value pairs are modeled as an edge (key) pointing to a node (value).  
- Arrays use `:INDEX` edges with a numeric `index` to link to values.  

**Description Guidelines:**  
0. First sentence of your description should explain what information is being retrieved from the path. Second sentence should be a special note about the array and objects and their indices/children as shown in examples below. and Third sentence should be a pedantic walk through each node and relationship. 
1. Include key names (`name` property), relationship types (`:DATASET`, `:FILES`, `:INDEX`), node labels (`:Pennsieve`, `:Dataset`, `:File`, `:Data`), and significant properties (`children`, `type`, `value`, `index`) to allow accurate query matching.
5. Keep your descriptions to 4-5 sentences maximum. Do not respond with anything else other than the description (no apologies or additional text at all).
6. Avoid filler language, use keywords from the path instead. Don't repeat information unnecessarily.

Examples paths and their explanations:
{
0. path: (:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_rawSignals]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 600.0, type: 'Array'})-[:INDEX]->(:Data {children: 200.0, type: 'Object'})-[:`5`]->(:Data {value: -3112.0})
Description: Retrieves a raw signal measurement (value: -3112.0) from the EDF file test.edf in dataset Test Dataset CNT. 
Special note: the key :_rawSignals returns an Array with 12 children (indices 0–11); within this array, an :INDEX relationship navigates to an inner Array of 600 children and then to an Object with 200 key–value pairs, where numeric index 5 selects the measurement. 
Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'Test Dataset CNT'), then :FILES accesses the :File node (name: 'test.edf'); a :DATA relationship retrieves an Object node (children: 3.0, type: 'Object'), from which the :_rawSignals key retrieves an Array node (children: 12.0, type: 'Array'); subsequent :INDEX relationships navigate an Array node (children: 600.0, type: 'Array') and then an Object node (children: 200.0, type: 'Object') before the numeric index 5 retrieves the leaf node (value: -3112.0)


1. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'A mathematical model for simulating the neural regulation'})-[:FILES]->(:File {name: 'manifest.json'})-[:DATA]->(:Data {children: 19.0, type: 'Object'})-[:creator]->(:Data {children: 3.0, type: 'Object'})-[:first_name]->(:Data {value: 'Omkar'})"
Description: Retrieves the creator’s first name ('Omkar') from the manifest.json file in dataset A mathematical model for simulating the neural regulation. Special note: the manifest.json file’s :Data node is an Object with 19 children; its :creator key returns an Object with 3 children, from which the :first_name key directly retrieves the leaf value 'Omkar'. Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'A mathematical model for simulating the neural regulation'), then :FILES accesses the :File node (name: 'manifest.json'); a :DATA relationship retrieves an Object node (children: 19.0, type: 'Object'), from which the :creator key accesses an Object node (children: 3.0, type: 'Object') and the :first_name key retrieves the leaf node (value: 'Omkar').

2. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_physicalSignals]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 600.0, type: 'Array'})-[:INDEX]->(:Data {children: 200.0, type: 'Object'})-[:`0`]->(:Data {value: 99.99237})"
Description: Retrieves a physical signal measurement (value: 99.99237) from the EDF file test.edf in dataset Test Dataset CNT. Special note: the key :_physicalSignals returns an Array with 12 children (indices 0–11); subsequent :INDEX relationships traverse an inner Array with 600 children and an Object with 200 key–value pairs, with numeric index 0 selecting the measurement. Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'Test Dataset CNT'), then :FILES accesses the :File node (name: 'test.edf'); a :DATA relationship retrieves an Object node (children: 3.0, type: 'Object'), from which the :_physicalSignals key retrieves an Array node (children: 12.0, type: 'Array'); subsequent :INDEX relationships navigate an Array node (children: 600.0, type: 'Array') and then an Object node (children: 200.0, type: 'Object') before the numeric index 0 retrieves the leaf node (value: 99.99237).

3. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_header]->(:Data {children: 10.0, type: 'Object'})-[:nbSignals]->(:Data {value: 12.0})"
Description: Retrieves the number of signal channels (value: 12.0) from the header of the EDF file test.edf in dataset Test Dataset CNT. Special note: the key :_header returns an Object with 10 key–value pairs, where the :nbSignals key directly retrieves the number of channels. Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'Test Dataset CNT'), then :FILES accesses the :File node (name: 'test.edf'); a :DATA relationship retrieves an Object node (children: 3.0, type: 'Object'), from which the :_header key retrieves an Object node (children: 10.0, type: 'Object') and the :nbSignals key retrieves the leaf node (value: 12.0).

4. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_header]->(:Data {children: 10.0, type: 'Object'})-[:signalInfo]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 10.0, type: 'Object'})-[:digitalMinimum]->(:Data {value: -32768.0})"
Description: Retrieves the digital minimum value (value: -32768.0) from the signal information in the header of the EDF file test.edf in dataset Test Dataset CNT. Special note: within the :_header object (which has 10 key–value pairs), the :signalInfo key returns an Array with 12 children (indices 0–11), where each element is an Object; an :INDEX relationship selects the specific object that contains the :digitalMinimum key. Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'Test Dataset CNT'), then :FILES accesses the :File node (name: 'test.edf'); a :DATA relationship retrieves an Object node (children: 3.0, type: 'Object'), from which the :_header key retrieves an Object node (children: 10.0, type: 'Object'), then the :signalInfo key retrieves an Array node (children: 12.0, type: 'Array'), followed by an :INDEX relationship accessing an Object node (children: 10.0, type: 'Object') where the :digitalMinimum key retrieves the leaf node (value: -32768.0).

5. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'A mathematical model for simulating the neural regulation'})-[:FILES]->(:File {name: 'manifest.json'})-[:DATA]->(:Data {children: 19.0, type: 'Object'})-[:license]->(:Data {value: 'Creative Commons Attribution'})" 
Description: Retrieves the license information ('Creative Commons Attribution') from the manifest.json file in dataset A mathematical model for simulating the neural regulation. Special note: the manifest.json file’s :Data node is an Object with 19 children, and its :license key directly retrieves the licensing value. Starting at :Pennsieve, the :DATASET relationship leads to a :Dataset node (name: 'A mathematical model for simulating the neural regulation'), then :FILES accesses the :File node (name: 'manifest.json'); a :DATA relationship retrieves an Object node (children: 19.0, type: 'Object') from which the :license key retrieves the leaf node (value: 'Creative Commons Attribution').
}

In your output provide only the textual description without any additional context or explanations. Max 4-5 lines.
"""


def generate_path_descriptions(all_paths: List[str]) -> List[str]:
    """
    Generate descriptions for a list of Cypher paths using OpenAI API.
    Note: The output descriptions are in the same order as the input Cypher paths.
    For example, the description at index 0 in the output corresponds to the Cypher path at index 0 in the input.

    Args:
        all_paths: List[str]: List of Cypher paths to generate descriptions for.

    Returns:
        List[str]: List of descriptions corresponding to the input paths.
    """
    # Load environment variables
    load_dotenv()

    # Load OpenAI API connection
    api_key = os.environ['OPENAI_API_KEY']
    if api_key:
        # descriptions for all input paths
        results: List[str] = []

        # Initialize ChatOpenAI
        chat = ChatOpenAI(
            model="o1-mini-2024-09-12",
            temperature=1,
            timeout=None,
            max_retries=2,
            )
        # Create a SystemMessage object
        system_message = SystemMessage(content=system_message_str)

        # iterate through each path in input
        for path in all_paths:
            # Create a HumanMessage object
            combined_content = f"{system_message.content}\n\nPath:{path}"
            human_message = HumanMessage(content=combined_content)
            # Invoke the OpenAI API on Human and System messages
            response = chat.invoke([human_message])
            # Append the response to the results list
            results.append(response.content)
        return results
    else:
        print("OpenAI API key is not set. Please check your .env file.")
        return []


def generate_embedding(path_description: str) -> List[float]:
    """
    Generate a vector embedding for a given path description using OpenAI API.
    Note: if you change the dimension here, make sure to change the dimension into `embedding` field in vectorDB

    Args:
        path_description (str): Description of a Cypher path to generate an embedding for.

    Returns:
        List[float]: Embedding vector for the path description.
    """
    # Load environment variables
    load_dotenv()

    # Test OpenAI API connection
    api_key = os.environ['OPENAI_API_KEY']
    if api_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key, dimensions=512)
        # Generate embedding for the path description
        embedding = embeddings.embed_query(path_description)
        return embedding
    else:
        raise Exception("OpenAI API key was not found. Please check your .env file.")


