from qa_chain import run_query
import json
import time

user_query1 = "Does dataset named: Test Dataset CNT has banner.jpg file?"
user_query2 = "What are the last names of contributors in dataset named Test Dataset CNT?"
user_query3 = "What is the degree (i.e. bachelors/masters/phd) of the contributors in our database?"
user_query4 = "What are the names of the datasets in our database?"
user_query5 = "What is the patientId of the patient in the header of the test.edf file in dataset named Test Dataset CNT?"

user_query5i = "What license does the dataset about mathematical model for simulating neural regulation have? This info is usually in the manifest.json file"

user_query6 = "What kind of files are in the dataset about mathematical model for simulating neural regulation"

user_query7 = "Who is the creator of the dataset about mathematical model for simulating neural regulation? look in the manifest.json file for this"

user_query8 = "Give me the orcids of the contributors of our datasets"

user_query9 = "Give me the name of the contributors or creators of the datasets in our database along with their orcids and the names of the datasets they contributed to"

user_query10 = "give me 10 values for raw signals in the edf file of Test Dataset CNT?"

start_time = time.time()
response = run_query(user_query10)  # change the user_query to test different questions
end_time = time.time()

# print(json.dumps(response, indent=2))  # prints all the intermediate steps and final answer
# print(f"\nFinal LLM answer: {response['result']}")  # prints only the final english answer that LLM generates after all the processing
# print(f"*******Time taken to execute CypherQAChain: {end_time - start_time:.3f} seconds")

print(response)
