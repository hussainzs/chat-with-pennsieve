# streamlit_app.py
import streamlit as st
import pandas as pd
import json
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
st.markdown('<div class="header-container"><div class="header">Pennsieve Query Engine</div></div>', unsafe_allow_html=True)

# Use columns to center the input and button (input box will be in the middle column, ~40% width)
cols = st.columns([3, 4, 3])
with cols[1]:
    user_query = st.text_area("", placeholder="Enter your query", height=100)
    submit = st.button("Submit Query")

if submit:
    if user_query.strip():
        with st.spinner("Processing your query please wait..."):
            response = process_query(user_query)

        st.subheader("Results")

        # 1. Final LLM Answer
        final_answer = response.get("result", "No result returned")
        st.markdown(f'<div class="result-box"><strong>Final LLM answer:</strong><br>{final_answer}</div>', unsafe_allow_html=True)

        # 2. Generated Cypher (Heading outside the box)
        generated_cypher = "No generated Cypher found"
        intermediate_steps = response.get("intermediate_steps", [])
        if intermediate_steps:
            if "query" in intermediate_steps[0]:
                generated_cypher = intermediate_steps[0]["query"]
                # remove the starting "cypher\n" from the generated cypher
                generated_cypher = generated_cypher.replace("cypher\n", "") if generated_cypher.startswith("cypher\n") else generated_cypher
            else:
                generated_cypher = "Cypher query couldn't be parsed properly"
        st.markdown("#### Generated Cypher", unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{generated_cypher}</div>', unsafe_allow_html=True)

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
