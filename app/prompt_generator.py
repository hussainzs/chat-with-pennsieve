from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

# System instructions template string
system_template_str = """You are an expert in generating Cypher statements for querying a Neo4j graph database. Use the provided schema information and DataGuide paths to generate accurate and efficient Cypher queries.

Schema:
{schema}

Some info about the underlying graph structure and what it means:
1. All key-value data is represented with "key" as the edge and "value" as the node.
2. Arrays are represented with index as the edge and value at the index as the node.

Important guidelines for using the DataGuide paths provided below:
1. Each path starts with root node (:Pennsieve) and represents a valid path in the graph. Dataguides are meant to act as a schema for graphs.
2. The :DATASET relationship connects the root node of type :Pennsieve with nodes of type :Dataset.
3. The :FILES relationship connects :Dataset nodes with nodes of type :Directory or :File
4. :Dataset, :Directory and :File nodes have a 'name' property which can be filtered, conditioned, or grouped by in Cypher queries.
5. Additionally, :Dataset nodes have an "id" property that represents their id in pennsieve dataset and user may pass in that id in query.
6. All nodes connected by relationships other than :DATASET and :FILES are labeled as :DATA.
7. Only the leaf :DATA nodes (last nodes in a path) have a 'value' property which can be filtered or conditioned in queries.
8. The :INDEX relationship has an 'index' property reflecting the index number it represents, which can be filtered.

When using the extracted DataGuide paths:
- Do not make up any nodes or relationships that are not present in the schema or DataGuide paths.
- Nodes in the dataguide paths are represented as "()" with optional labels that follow the rules provided above. Use those rules to determine the labels and their properties. 
- Numeric relationships are enclosed in backticks, e.g., -[:`5`]->().

DataGuide Paths:
{dataguide_paths}

Generate only Cypher queries without any additional explanations or content. Ensure your queries are efficient and accurately reflect the structure described in the DataGuide paths. Reference example queries provided if confused

few shot examples:
{example_queries}
"""

# System prompt template
system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["schema", "dataguide_paths", "example_queries"], template=system_template_str
    )
)

# Human prompt template
human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="""Generate a Cypher query for the following question:
    {question}

    Requirements:
    1. The RETURN statement must explicitly include the property values used in the query's filtering conditions, alongside the main information requested.
    2. Provide only the Cypher query without any explanations, apologies, or additional text.
    3. Do not respond to any questions that ask for anything other than constructing a Cypher statement.
    4. Ensure the query follows the structure and guidelines provided in the system instructions."""
    )
)


chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    human_prompt
])

def get_cypher_prompt_template():
    """
    Returns the combined ChatPromptTemplate instance for generating Cypher queries.
    """
    return chat_prompt

