# Pennsieve Graph Layer Project

## Project Description
This is the research project component developed under the guidance of Dr. Zachary Ives. The initial goal is to develop a graph layer on top of the Pennsieve database and enable machine learning through effective data extraction of medical data from complex and versatile file formats.

**Note**: All methods were implemented on the underlying graph built on Neo4j using another repository which will be linked once it is public.

## Project Structure

### app/
- **`__init__.py`**: Initializes the app package.
  - **Purpose**: Marks the directory as a Python package. Add package-level imports here if needed.
- **`config.py`**: Handles configuration and environment variables.
  - **Purpose**: Loads environment variables and defines configuration settings.
  - **Enhancements**: Implement error handling for missing environment variables if needed.
- **`database.py`**: Manages Neo4j database connection.
  - **Purpose**: The function `setup_neo4j_graph()` returns a Neo4j graph configured with URL, username, and password provided in the `.env` file.
  - **Documentation**: `setup_neo4j_graph()` returns the Langchain Neo4j database wrapper. Important methods used: `query()` and `refresh_schema()`. [Neo4jGraph Documentation](https://api.python.langchain.com/en/latest/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html)
- **`dataguide.py`**: Extracts dataguide paths from the database and formats them into Cypher paths.
  - **Methods**:
    1. `extract_dataguide_paths(graph: Neo4jGraph)`: Extracts dataguide paths from root to leaf using a Cypher query.
    2. `format_paths_for_llm(results: List[Dict[str, Any]])`: Formats results from `extract_dataguide_paths` into valid Cypher paths for MATCH queries.
- **`test.py`**: Tests the connection with Neo4j graph, extraction of dataguide paths, and formatting them. Outputs the time taken for each part.
  - **Enhancements**: Add unit testing or test other methods manually.
- **`prompt_generator.py`**: Creates and combines Langchain system and human prompts into `langchain.prompts.ChatPromptTemplate`.
  - **Methods**:
    - `get_cypher_prompt_template()`: Returns the `ChatPromptTemplate` instance.
  - **Documentation**:
    - [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
    - [SystemMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.SystemMessagePromptTemplate.html)
    - [HumanMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.HumanMessagePromptTemplate.html)
    - [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- **`qa_chain.py`**: Defines the `run_query(user_query: str)` function, which integrates all project components and runs a `GraphCypherQAChain` on the user query.
  - **Documentation**:
    - [GraphCypherQAChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.graph_qa.cypher.GraphCypherQAChain.html)
    - [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
    - Note: Replace `ChatOpenAI` with [AzureChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html) if needed.
- **`main.py`**: Passes the user query and retrieves the result by calling `run_query(user_query: str)` from `qa_chain.py`.

### paths_vectorDB/
- **`__init__.py`**: Initializes the app package.
  - **Purpose**: Marks the directory as a Python package. Add package-level imports here if needed.
- **`generate_descriptions.py`**: Defines the system prompt to generate descriptions from LLMs for Cypher paths. Provides methods to generate a list of path descriptions and produce vector embeddings for user queries.
  - **Documentation**: [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- **`random_path_generator.py`**: Provides methods to generate random paths from the database and format them into Cypher paths.
- **`vectorDB_setup.py`**: Methods to start Milvus container, connect with it, define collection schema, create collection, insert data, and conduct vector similarity searches.
  - **Documentation**: [pymilvus](https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Collections/create_collection.md)
- **`main.py`**: Wrapper functions that combine all functionalities from this directory. For example, `get_similar_paths_from_milvus` is used in `app/qa_chain.py` to conduct vector similarity search with user queries.
- **`test.py`**: Methods to test various functionalities. Currently commented out.
  - **Enhancements**: Add unit testing or test other methods manually.

### Root Directory
- **`write_read_data.py`**: Simple write and read methods to store Cypher paths and descriptions generated from API calls. Helps with analysis and saving API costs. The method `fill_collection_with_random_paths` in `paths_vectorDB/main.py` writes down the paths and descriptions generated from API calls.
- **`env.sample`**: Make a copy of this in your project root directory and rename it to `.env`. Fill in the values.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`README.md`**: Project documentation.
- **`docker-compose.yml`**: Docker file for Milvus DB. If there is a new version, replace this file. Ensure it is named `docker-compose.yml` and placed in the root directory.
- **`requirements.txt`**: Python dependencies and their compatible versions used for development. Note: The `requirements.txt` file was created through `pipenv`.

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Neo4j Desktop

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd project_root

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `env.sample` to `.env` and fill in the required values.

4. **Set up Neo4j Desktop**:
   - Download and install [Neo4j Desktop](https://neo4j.com/download/).
   - Create a new project and database. Note the URL, username, and password.
   - **Note**: FILL THE NEO4J database with the correct data using the scripts provided in the other repository (will link soon)
   - Update the `.env` file with the Neo4j connection details (URL, username, password).

5. **Set up Milvus**:
   - Start Milvus using Docker:
     ```bash
     docker-compose up -d
     ```
   - Follow the steps in `paths_vectorDB/vectorDB_setup.py` to connect to Milvus and set up the database.

6. **Populate the database**:
   - Run the script to populate the Neo4j database with the correct data.
     ```bash
     python scripts/setup_database.py
     ```

## Recommended Enhancements
1. **Improve System Prompts**: Enhancing the prompts in both `app` and `paths_vectorDB` can significantly improve LLM performance.
2. **Optimize Path Selection**: Instead of sending all dataguide paths, send the top 10 related paths from the Milvus vector DB to reduce API costs and potentially improve performance.
3. **Update Milvus**: Install the latest version of Milvus and change the similarity metric from "IP" (Inner Product) to COSINE.
4. **Create a UI**: Use Streamlit to create a basic user interface and FastAPI to create an API for interaction.
5. **Add Conversational Ability**: Allow for follow-up interactions to guide the LLM for better path generation, although this may increase API costs.


