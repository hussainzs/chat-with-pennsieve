from qa_chain import run_query

user_query1 = "Does dataset named: Test Dataset CNT has banner.jpg file?"
user_query2 = "What are the last names of contributors in dataset named Test Dataset CNT?"
user_query3 = "What is the degree of the contributors in our database?"
response = run_query(user_query2)
print(response)