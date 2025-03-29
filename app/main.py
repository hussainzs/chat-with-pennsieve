from app.qa_chain import run_query
import time
import json

example_query1 = "Does dataset named: Test Dataset CNT has banner.jpg file?"
example_query2 = "What are the last names of contributors in dataset named Test Dataset CNT?"
example_query3 = "What is the degree (i.e. bachelors/masters/phd) of the contributors in our database?"
example_query4 = "What are the names of the datasets in our database?"
example_query5 = "What is the patientId of the patient in the header of the test.edf file in dataset named Test Dataset CNT?"

example_query5i = "What license does the dataset about mathematical model for simulating neural regulation have? This info is usually in the manifest.json file"

example_query6 = "What kind of files are in the dataset about mathematical model for simulating neural regulation"

example_query7 = "Who is the creator of the dataset about mathematical model for simulating neural regulation? look in the manifest.json file for this"

example_query8 = "Give me the orcids of the contributors of our datasets"

example_query9 = "Give me the name of the contributors or creators of the datasets in our database along with their orcids and the names of the datasets they contributed to"

example_query10 = "give me 10 values for raw signals in the edf file of Test Dataset CNT?"


def process_query(user_query: str) -> dict:
    """
    Given a user_query, run the Langchain CypherQA chain
    and return the full response.
    """
    response: dict = run_query(user_query)
    return response


if __name__ == '__main__':
    # If running from terminal, allow passing query via command-line argument
    import sys

    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        start_time = time.time()
        chain_response = run_query(user_query)  # change the user_query to test different questions
        end_time = time.time()

        print(json.dumps(chain_response, indent=2))  # prints all the intermediate steps and final answer
        print(
            f"\nFinal LLM answer: {chain_response['result']}")  # prints only the final english answer that LLM generates after all the processing
        print(f"*******Time taken to execute CypherQAChain: {end_time - start_time:.3f} seconds")
    else:
        print("Please provide a query as a command-line argument.")
