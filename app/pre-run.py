from app.main import process_query

if __name__ == '__main__':
    print("This script is intended to be run before running streamlit app")
    print("It is used to pre-fill the vector DB with paths and their descriptions.")
    print("This is useful for the first time setup or if the collection needs to be rebuilt.\n")
    print("Running pre-run script...")
    try:
        process_query("What are the names of the datasets in our database?")
        print("Pre-run script completed successfully.")
    except Exception as e:
        print(f"ERROR: An error occurred while running the pre-run script: {e}")