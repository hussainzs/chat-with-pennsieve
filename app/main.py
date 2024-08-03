from qa_chain import run_query
import json
import time

user_query1 = "Does dataset named: Test Dataset CNT has banner.jpg file?"
user_query2 = "What are the last names of contributors in dataset named Test Dataset CNT?"
user_query3 = "What is the degree of the contributors in our database?"
user_query4 = "What are the names of the datasets in our database?"
user_query5 = "What is the patientId of the patient in the header of the test.edf file in dataset named Test Dataset CNT?"

start_time = time.time()
response = run_query(user_query5)  # change the user_query to test different questions
end_time = time.time()


print(json.dumps(response, indent=2))  # prints all the intermediate steps and final answer
print(f"\nFinal LLM answer: {response['result']}")  # prints only the final english answer that LLM generates after all the processing
print(f"*******Time taken to execute CypherQAChain: {end_time - start_time:.3f} seconds")