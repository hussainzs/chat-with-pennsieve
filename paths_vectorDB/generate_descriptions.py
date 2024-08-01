from langchain.schema import HumanMessage, SystemMessage
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# System instructions template string
system_message_str = """
You are a Neo4j expert specializing in medical datasets and files. Given a Cypher path, your task is to provide a concise description of the specific information this path can retrieve. Your description should be tailored to match similar user queries using vector similarity.

Guidelines for your descriptions:
1. Focus on the key elements of the path: dataset names, file names, and specific properties of nodes.
2. Mention Relationship names and Node labels to provide context. Also mention value property of :DATA nodes (especially mention the value property of the last node of the path)
3. :DATA nodes, often have children property and type property. Children property describes number of elements in the data structure and type property explains the type of data structure i.e.type: Object (Dictionary) or Array
4. Highlight the significance of :INDEX relationships especially the numeric edges in accessing specific data points. 
5. Pay attention to file extensions and use domain-specific terminology related to the common structure of such files i.e. EDF files have headers with meta data and signal data.
6. Keep your description to a maximum of 3 sentences, prioritizing clarity and specificity.
7. Avoid filler words instead Incorporate key words from the path to enhance matching with user queries.

Structure of Graph and paths:
- The root node is always (:Pennsieve).
- :DATASET relationship connects to :Dataset nodes, :FILES relationship connects to :File or :Directory nodes.
- Nodes after :FILES are labeled :DATA, representing file contents.
- Key-value data within files is mapped onto graph as Key -> relationship name and Value -> node value
- Array data has index as relationship name and value at index as node value

Examples paths and their explanations:
{
0. path: (:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_rawSignals]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 600.0, type: 'Array'})-[:INDEX]->(:Data {children: 200.0, type: 'Object'})-[:`5`]->(:Data {value: -3112.0})
Description 2: Retrieve the raw_signal data from the 'test.edf' file in 'Test Dataset CNT'. This path accesses nested array data to access a specific value (-3112.0) at index 5.

1. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'A mathematical model for simulating the neural regulation'})-[:FILES]->(:File {name: 'manifest.json'})-[:DATA]->(:Data {children: 19.0, type: 'Object'})-[:creator]->(:Data {children: 3.0, type: 'Object'})-[:first_name]->(:Data {value: 'Omkar'})"
answer: "Retrieve the first name of the creator of dataset named 'A mathematical model for simulating the neural regulation'. This path accesses nested object data within the manifest.jspn file."

2. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_physicalSignals]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 600.0, type: 'Array'})-[:INDEX]->(:Data {children: 200.0, type: 'Object'})-[:`0`]->(:Data {value: 99.99237})"
answer: "Access physical signal value of 99.99237 data from the 'test.edf' file in 'Test Dataset CNT'. This path retrieves a precise value (99.99237) from _physicalSignals nested array structure representing 12 channels of 600 sample points each located at index 0"

3. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_header]->(:Data {children: 10.0, type: 'Object'})-[:nbSignals]->(:Data {value: 12.0})"
answer: "Retrieve the number of signals (12) from the header of 'test.edf' in 'Test Dataset CNT'. This path accesses metadata in the header, specifically nbSignals value."

4. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'Test Dataset CNT'})-[:FILES]->(:File {name: 'test.edf'})-[:DATA]->(:Data {children: 3.0, type: 'Object'})-[:_header]->(:Data {children: 10.0, type: 'Object'})-[:signalInfo]->(:Data {children: 12.0, type: 'Array'})-[:INDEX]->(:Data {children: 10.0, type: 'Object'})-[:digitalMinimum]->(:Data {value: -32768.0})"
answer: "Access the digital minimum value of -32768.0 from signalInfo from the 'test.edf' file in 'Test Dataset CNT'. This path navigates through the file's header to retrieve signal information. Header is object with 10 children and signalInfo is an array with 12 elements within it."

5. path: "(:Pennsieve)-[:DATASET]->(:Dataset {name: 'A mathematical model for simulating the neural regulation'})-[:FILES]->(:File {name: 'manifest.json'})-[:DATA]->(:Data {children: 19.0, type: 'Object'})-[:license]->(:Data {value: 'Creative Commons Attribution'})"
answer: "Retrieve the license information ('Creative Commons Attribution') for the dataset 'A mathematical model for simulating the neural regulation'. This path accesses license metadata from the manifest.json file."
}

Provide only the description without any additional context or explanations. Max 3 lines.
"""


def generate_path_descriptions(all_paths: List[str]) -> List[str]:
    """
    Generate descriptions for a list of Cypher paths using OpenAI API and a pre-defined system prompt.
    Note: The output descriptions are in the same order as the input Cypher paths.
    For example, the description at index 0 in the output corresponds to the Cypher path at index 0 in the input.

    Args:
        all_paths (List[str]): List of Cypher paths to generate descriptions for.

    Returns:
        List[str]: List of descriptions corresponding to the input paths.
    """
    # Load environment variables
    load_dotenv()

    # Test OpenAI API connection
    api_key = os.environ['OPENAI_API_KEY']
    if api_key:
        # this will contain descriptions for all paths
        results: List[str] = []

        # Initialize ChatOpenAI
        chat = ChatOpenAI(
            model_name="gpt-4o-2024-05-13",
            temperature=0,
            openai_api_key=api_key
        )
        # Create a SystemMessage object
        system_message = SystemMessage(content=system_message_str)

        # iterate through each path in input
        for path in all_paths:
            # Create a HumanMessage object
            human_message = HumanMessage(content=path)
            # Invoke the OpenAI API on Human and System messages
            response = chat.invoke([human_message, system_message])
            # Append the response to the results list
            results.append(response.content)
        return results
    else:
        print("OpenAI API key is not set. Please check your .env file.")
        return []


def generate_embedding(path_description: str) -> List[float]:
    """
    Generate an embedding for a given path description using OpenAI API.

    Args:
        path_description (str): Description of a Cypher path.

    Returns:
        List[float]: Embedding vector for the path description.
    """
    # Load environment variables
    load_dotenv()

    # Test OpenAI API connection
    api_key = os.environ['OPENAI_API_KEY']
    if api_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key, dimensions=512)
        # Generate embedding for the path description
        embedding = embeddings.embed_query(path_description)
        return embedding
    else:
        print("OpenAI API key is not set. Please check your .env file.")
        return []


