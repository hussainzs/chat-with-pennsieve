# Chat With [Pennsieve](https://app.pennsieve.io/)



[demo](https://github.com/user-attachments/assets/b2ea175f-6637-4744-9093-dd75c39fac1b)



# TODO: will be updated soon

## Project Description
This is the research project component developed under the guidance of [Dr. Zachary Ives](https://www.cis.upenn.edu/~zives/). The initial goal is to develop a graph layer on top of the [Pennsieve database](https://app.pennsieve.io/) and enable machine learning through effective data extraction of medical data from complex and versatile file formats. This component **enables natural language interaction with the database.** 

**Note**: All methods were implemented on the underlying graph built on Neo4j using another repository which will be linked once it is public. This project is ready to be used out of the box, however, without the underlying graph filled in you will not get any results.

## Project Structure

### app/
- **`__init__.py`**: Initializes the app package.
  - **Purpose**: Marks the directory as a Python package. Add package-level imports here if needed.
- **`config.py`**: Handles configuration and environment variables.
  - **Purpose**: Loads environment variables and defines configuration settings.
  - **Enhancements**: Implement error handling for missing environment variables if needed.
- **`database.py`**: Manages Neo4j database connection.
  - **Purpose**: The function `setup_neo4j_graph()` returns a Neo4j graph configured with URL, username, and password provided in the `.env` file.
  - **Documentation**: `setup_neo4j_graph()` returns the Langchain Neo4j database wrapper. Important methods used: `query()` and `refresh_schema()`. [Langchain Neo4jGraph Documentation](https://api.python.langchain.com/en/latest/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html)
- **`main.py`**: Entry point of the application. Pass the user query and retrieves the result by calling `run_query(user_query: str)` from `qa_chain.py`. It abstracts away all the complexities and provides a simple interface to interact with the system.
- **`dataguide.py`**: Extracts dataguide paths from the database and formats them into Cypher paths.
  - **Methods**:
    1. `extract_dataguide_paths(graph: Neo4jGraph)`: Extracts dataguide paths from root to leaf using a Cypher query.
    2. `format_paths_for_llm(results: List[Dict[str, Any]])`: Formats results from `extract_dataguide_paths` into valid Cypher paths for MATCH queries.
- **`test.py`**: Tests the connection with Neo4j graph, extraction of dataguide paths, and formatting them. Outputs the time taken for each part.
  - **Enhancements**: Add unit testing or test other methods manually.
- **`prompt_generator.py`**: This module is responsible for creating and combining Langchain _system_ and _human_ prompts into `langchain.prompts.ChatPromptTemplate`. It is a crucial part of the project as it defines how the prompts are structured and used in the Langchain framework.
  - **Methods**:
    - `get_cypher_prompt_template()`: This method returns the `ChatPromptTemplate` instance created in this file. It combines system and human prompts into a single template that can be used to generate Cypher queries from `GraphCypherQAChain` in `qa_chain.py`.
  - **Documentation**:
    - [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html): This class is used to define the structure of the prompts. The primary parameters used are `input_variables`, which specify the variables to be included in the prompt, and `template`, which defines the prompt's text.
    - [SystemMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.SystemMessagePromptTemplate.html): This class is used to create system messages in the prompt. The primary parameter used is `prompt`, which defines the system message's text.
    - [HumanMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.HumanMessagePromptTemplate.html): This class is used to create human messages in the prompt. The primary parameter used is `prompt`, which defines the human message's text.
    - [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html): This class combines system and human messages into a single chat prompt. The primary method used is `from_messages()`, which takes a list of message templates and combines them into a chat prompt.
- **`qa_chain.py`**: Defines the `run_query(user_query: str)` function, which integrates all project components and runs a `GraphCypherQAChain` on the user query.
  - **Documentation**:
    - [GraphCypherQAChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.graph_qa.cypher.GraphCypherQAChain.html)
    - [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
    - Note: Replace `ChatOpenAI` with [AzureChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html) if needed.
### paths_vectorDB/
- **`__init__.py`**: Initializes the app package.
  - **Purpose**: Marks the directory as a Python package. Add package-level imports here if needed.
- **`generate_descriptions.py`**: Defines the system prompt to generate descriptions from LLMs for Cypher paths. 
  - **Methods**:
    - `generate_path_descriptions(all_paths: List[str])`: Generates descriptions for the given paths using the LLM. Outputs a list of descriptions.
    - `generate_embedding(path_description: str)`: Generates embeddings for the given path description using the OpenAI embeddings API.
  - **Documentation**: [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- **`random_path_generator.py`**: Provides methods to generate random paths from the database and format them into Cypher paths.
- **`vectorDB_setup.py`**: Provides methods to start Milvus container, connect with it, define collection schema, create collection, insert data, and conduct vector similarity searches.
  - **Documentation**: [pymilvus](https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Collections/create_collection.md)
- **`main.py`**: Wrapper functions that combine all functionalities from this directory. For example, `get_similar_paths_from_milvus` is used in `app/qa_chain.py` to conduct vector similarity search with user queries.
- **`test.py`**: Methods to test various functionalities. Currently commented out.
  - **Enhancements**: Add unit testing or test methods manually.
- **`write_read_data.py`**: Simple write and read methods to store Cypher paths and descriptions generated from API calls. 
  - **Purpose**: Helps with analysis and saving API costs. The method `fill_collection_with_random_paths` in `paths_vectorDB/main.py` writes down the paths and descriptions generated from API calls into `data.txt`.

### Root Directory
- **`env.sample`**: Make a copy of this in your project root directory and rename it to `.env`. Fill in the values.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`README.md`**: Project documentation.
- **`docker-compose.yml`**: Docker file for Milvus DB. If there is a new version, replace this file. Ensure it is named `docker-compose.yml` and placed in the root directory.
- **`requirements.txt`**: Python dependencies and their compatible versions used for development. Note: The `requirements.txt` file was created through `pipenv`.

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Neo4j Desktop and Neo4j Database filled with graph and dataguide _(code for this will be linked soon)_

### Installation
Getting started with this project is simple. You can follow the steps below:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/hussainzs/chat-with-pennsieve.git
   cd project_root
   ``` 
   **note:** Make sure you are in the project root directory before proceeding with the next steps.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `env.sample` and rename the file to `.env` and fill in the required values.

4. **Set up Neo4j Desktop**:
   - Download and install [Neo4j Desktop](https://neo4j.com/download/).
   - Note the URL, username, and password for the Neo4j database that contains the graph and dataguide.
   - Update the `.env` file with the Neo4j connection details (URL, username, password). Default values have been filled in.

5. **Run app/main.py**:
   - Navigate to the `app` directory and run `main.py`. Make sure your desired user query is passed as an argument to the `run_query(user_query)` function.
   - Make sure you have `docker-compose.yml` in the root directory. When you run app/main.py, the Milvus containers will start automatically by running terminal commands. Check out `paths_vectorDB/vectorDB_setup.py` for more information.
   - **Note**: When the Milvus container is created the first time, it downloads  and creates a new folder in the root directory named `volumes`. The folder contains 3 subfolders: `milvus`, `minio`, and `etcd`.
   - For more information check out: [Run Milvus using Docker Compose](https://milvus.io/docs/install_standalone-docker-compose.md)

**Note**: For further clarification of expected output when you run `app/main.py`, I'm attaching 2 pdfs of output generated from the system in the folder called **Expected Outputs**. 
1. The file named `first_output.pdf` shows what's expected when user runs the `app/main.py` for the first time in a new session with default values. (When you run it for the first time ever, it may take a while to download everything)
2. The `regular_output.pdf` shows what's expected when user runs the `app/main.py` in a regular session with default values.

## Recommended Enhancements
1. **Improve System Prompts**: Enhancing the prompts in both `app` and `paths_vectorDB` can significantly improve LLM performance. I witnessed that high quality examples in system prompt will increase the quality of description generation for paths. System Prompt also significantly effects the final answer from LLM.
2. **Optimize Context for LLM**: Instead of sending all dataguide paths, send the top 10 related paths from the Milvus vector DB to reduce API costs and potentially improve performance. Long system prompts can increase hallucination and confuses LLM, refer to this paper for more information: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172)
3. **Update Milvus**: Install the latest version of Milvus and change the similarity metric from "IP" (Inner Product) to COSINE in `search_similar_vectors` method inside of `paths_vectorDB/vectorDB_setup.py` for better results.
4. **Create a chat UI**: Use Streamlit or your favorite UI library to create a basic user interface for this project. You can use FastAPI to create a simple API for sending user queries and receiving responses from `app/main.py.`
5. **Add Conversational Ability**: Allow for follow-up interactions to guide the LLM for better path generation, although this may increase API costs. I noticed that often when LLM was wrong, it was only off by a little in its path generation. Someone with domain knowledge of the underlying graph can easily correct it with a basic follow-up.

