from qa_chain import run_query

user_query = "Does dataset named: Test Dataset CNT has banner.jpg file?"
response = run_query(user_query)
print(response)