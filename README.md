# Chat With [Pennsieve](https://app.pennsieve.io/)



[demo](https://github.com/user-attachments/assets/b2ea175f-6637-4744-9093-dd75c39fac1b)

This is the research project component developed under the guidance of [Dr. Zachary Ives](https://www.cis.upenn.edu/~zives/). The goal is to develop a graph layer on top of the [Pennsieve database](https://app.pennsieve.io/) and enable machine learning through effective data extraction of medical data from complex and versatile file formats. This component **enables natural language interaction with the database.** 

## Tech Stack

- **Streamlit** -  For UI
- **OpenAI** -  For LLM-powered query generation. Uses the `o1-mini` model.
- **LangChain**: For Code Logic and design
- **Neo4j**: For storing the underlying data representing the Pennsieve database.
- **Milvus**: As a vector database for storing the index that we conduct RAG over.
