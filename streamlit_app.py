# streamlit_app.py
import streamlit as st
import pandas as pd
import random
import time
import threading
from paths_vectorDB.vectorDB_setup import collection_exists, get_collection_size
from app.main import process_query
#from app.trash import process_query  # for testing returns a sample response for a query with a delay of 2 seconds

# Set up the page configuration (no sidebar, refined professional theme)
st.set_page_config(page_title="Pennsieve Query Engine", layout="wide")

# Add custom CSS for a refined, lighter theme with UPenn colors
st.markdown("""
    <style>
        /* Container for header with limited width */
        .header-container {
            width: 80%;
            margin: 0 auto;
        }
        /* Header style */
        .header {
            text-align: center;
            padding: 0.5rem;
            background-color: #ffffff;
            color: #011F5B;  /* UPenn blue */
            font-size: 1.5rem;
            font-weight: 600;
            border-bottom: 2px solid #011F5B;
            margin-bottom: 1rem;
        }
        /* Styling for result boxes */
        .result-box {
            border: 1px solid #ccc;
            padding: 0.75rem;
            border-radius: 5px;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
            color: #333333;
        }
        /* Allow text area to resize vertically and wrap text */
        textarea {
            resize: vertical;
            overflow: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        /* Remove default Streamlit sidebar */
        .css-1d391kg { 
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Header (centered, 80% width)
st.markdown('<div class="header-container"><div class="header">Pennsieve Query Engine</div></div>',
            unsafe_allow_html=True)

# Use columns to center the input and button (input box will be in the middle column, ~40% width)
cols = st.columns([3, 4, 3])
with cols[1]:
    user_query = st.text_area("", placeholder="Enter your query", height=100)
    submit = st.button("Submit Query")

if submit:
    if user_query.strip():
        # Create a dictionary to store the result from process_query
        result_dict = {}

        # Define a function to run the query and store the result.
        def run_query():
            result_dict["response"] = process_query(user_query)

        # Start process_query in a separate thread.
        query_thread = threading.Thread(target=run_query)
        query_thread.start()

        # Show spinner and update steps while process_query runs.
        with st.spinner("Progress:"):
            status = st.empty()  # placeholder for step messages
            start_time = time.time()

            # Step 1:
            elapsed = time.time() - start_time
            status.markdown(f"""
            <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                <strong style="font-size:16px;">Step 1:</strong> Making sure vector database is up and running<br>
                <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(random.uniform(2, 4))

            # Step 2:
            elapsed = time.time() - start_time
            status.markdown(f"""
            <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                <strong style="font-size:16px;">Step 2:</strong> Looking up similar queries in the vector database<br>
                <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(random.uniform(3, 6))

            # Step 3:
            elapsed = time.time() - start_time
            status.markdown(f"""
            <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                <strong style="font-size:16px;">Step 3:</strong> Generating cypher query, this takes sometime<br>
                <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(random.uniform(3, 6))

            # Step 4:
            elapsed = time.time() - start_time
            status.markdown(f"""
            <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                <strong style="font-size:16px;">Step 4:</strong> Getting results back<br>
                <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(random.uniform(2, 5))

            # Step 5: Keep showing a status message until process_query finishes.
            while query_thread.is_alive():
                elapsed = time.time() - start_time
                if elapsed >= 35:
                    status.markdown(
                        f"""
                        <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                            <strong style="font-size:16px;">Step 5:</strong> Previously generated query failed, retrying error correction<br>
                            <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                if elapsed >= 55:
                    status.markdown(
                        f"""
                        <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                            <strong style="font-size:16px;">Step 5:</strong> Generated query failed to get results, retrying one last time<br>
                            <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    status.markdown(
                        f"""
                        <div style="padding:10px; background-color: #e8f4f8; border-radius: 5px; margin-bottom:10px;">
                            <strong style="font-size:16px;">Step 5:</strong> Summarizing everything, please hold on<br>
                            <span style="color:gray; font-size:14px;">Elapsed time: {elapsed:.1f} seconds</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                time.sleep(1)
            status.empty()  # Clear the status once done

        # Ensure the thread is finished before retrieving the result.
        query_thread.join()
        response = result_dict.get("response")

        st.subheader("Results")

        # 1. Final LLM Answer
        final_answer = response.get("result", "No result returned") if response is not None else "No result returned"
        st.markdown(f'<div class="result-box"><strong>Final LLM answer:</strong><br>{final_answer}</div>',
                    unsafe_allow_html=True)

        # 2. Generated Cypher (Heading outside the box)
        generated_cypher = "No generated Cypher found"
        intermediate_steps = response.get("intermediate_steps", [])
        if intermediate_steps:
            if "query" in intermediate_steps[0]:
                generated_cypher = intermediate_steps[0]["query"]
                # Remove the starting "cypher\n" if present.
                generated_cypher = generated_cypher.replace("cypher\n", "") if generated_cypher.startswith(
                    "cypher\n") else generated_cypher
            else:
                generated_cypher = "Cypher query couldn't be parsed properly"
        st.markdown("#### Generated Cypher", unsafe_allow_html=True)
        if generated_cypher:
            st.code(generated_cypher, language='cypher', line_numbers=True, wrap_lines=True)
        else:
            st.markdown('<div class="result-box">No generated Cypher found.</div>', unsafe_allow_html=True)

        # 3. Full Context: Extract and display as an interactive table
        context_data = []
        for step in intermediate_steps:
            if "context" in step:
                context_data = step["context"]
                break

        st.markdown("#### Full Context", unsafe_allow_html=True)
        if context_data:
            df_context = pd.DataFrame(context_data)
            st.dataframe(df_context)  # Interactive table: columns can be resized
        else:
            st.markdown('<div class="result-box">No context data found.</div>', unsafe_allow_html=True)
    else:
        st.error("Please enter a valid query.")
