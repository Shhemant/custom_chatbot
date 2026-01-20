#import necessary packages
import streamlit as st
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy.spatial.distance
from openai import OpenAI
import json


st.set_page_config(page_title="Search Agent", page_icon="🌍")
st.title("Ecoinvent Search Agent")

# load data
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(DATA_DIR, "data", "df_ecoinvent.parquet")

# Initialize Groq Client - don't  forget to save api key in .streamlit/secrets.toml (GROQ_API_KEY = "..")
if "GROQ_API_KEY" in st.secrets:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )
else:
    st.error("Missing GROQ_API_KEY in .streamlit/secrets.toml")
    st.stop()

# add logging to see what is happening behind the scenes in the console
logger = logging.getLogger("custom chatbot")
# ADD THIS BLOCK
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG if you want to see logger.debug() calls
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Ensures logs go to the terminal
)
# bert match function
def bert_match(query, df_subset, number_top_matches=10):
    """
    Match query against a subset of the global dataframe.
    
    Args:
        query (str): Search string.
        df_subset (pd.DataFrame): The filtered dataframe containing 'embedding' column.
        number_top_matches (int): Number of results to return.
    
    Returns:
        list[dict]: List of matching activity dictionaries.
    """

    logger.info('Initiating bert search')
    if df_subset.empty:
        logger.warning("bert_match called with empty dataframe.")
        return []

    # 1. Encode the query
    query_embedding = model.encode([query])

    # 2. Prepare Data Embeddings
    # The 'embedding' column likely contains lists or 1D arrays. 
    # We stack them into a single 2D numpy matrix for vectorised calculation.
    try:
        # np.stack is much faster than converting list-of-lists
        corpus_embeddings = np.stack(df_subset['embedding'].values)
    except Exception as e:
        logger.error(f"Error stacking embeddings from dataframe: {e}")
        return []

    # 3. Calculate Distances (Cosine)
    # cdist expects 2D arrays. query_embedding is (1, 384), corpus is (N, 384)
    distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]

    # 4. Sort and Extract Results
    # Get indices of the top N smallest distances
    # argsort returns indices that sort the array. We take the first N.
    top_indices = np.argsort(distances)[:number_top_matches]
    
    results = []
    for idx in top_indices:
        # Map the numpy index back to the DataFrame row
        # distinct from .loc (label index) vs .iloc (integer position)
        row = df_subset.iloc[idx]
        score = 1 - distances[idx] # Convert distance to similarity score
        
        # 5. Structure the Output
        #technosphere database
        results.append({
            "name": row.get("name"),
            "reference_product": row.get("reference product", None), 
            "product_information": row.get("product information", None)
        })
    logger.info(f'returning results {results}')
    return json.dumps(results)


# 2. CACHED RESOURCES (Load once, use many times)
# ---------------------------------------------------------
@st.cache_resource
def load_ai_engine():
    """Load the embedding model and database once."""
    logger.info("⏳ Loading Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # load ecoinvent data - extracted using brightway
    logger.info("⏳ Loading Data...")
    df_eco_new = pd.read_parquet(data_path)
    logger.info(f'loaded {data_path}')
    logger.info(f'shape: {df_eco_new.shape}')
    return model, df_eco_new

model, df_eco = load_ai_engine()

# tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search for environmental data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term"}
                },
                "required": ["query"],
            },
        },
    }
]

# 4. CHAT LOGIC
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a system message (hidden from UI) to guide the agent
    st.session_state.messages.append({
        "role": "system", 
        "content": "You are a helpful Assistant. Always search the database before answering factual questions. You can call the function multiple times (max 3) if necessary"
    })

# Display chat history (exclude system message)
for message in st.session_state.messages:
    if message["role"] != "system":
        if message.get("content"):
            st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask about background datasets such as steel, wind, or transport..."):
    
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        
        # Initial Call (Check for tools)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.messages,
            tools=tools,
            tool_choice="auto",
            temperature=0
        )
        response_msg = response.choices[0].message
        tool_calls = response_msg.tool_calls

        # If the Agent wants to use a tool:
        if tool_calls:
            # Convert to dict
            msg_dict = {
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            }
            st.session_state.messages.append(msg_dict)
            
            # Show a nice spinner while tool runs
            with st.status("Searching Database...", expanded=True) as status:
                for tool_call in tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    query = args["query"]
                    
                    status.write(f"Looking for '{query}'...")
                    raw_result = bert_match(query, df_eco)
                    
                    # 1. Handle None/Empty results explicitly
                    if raw_result is None or (isinstance(raw_result, pd.DataFrame) and raw_result.empty):
                        tool_result = "No matching data found in the database."
                    # 2. Convert DataFrames/Objects to string
                    else:
                        tool_result = str(raw_result)

                    st.session_state.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": tool_result
                    })
                    status.update(label="Data Found!", state="complete", expanded=False)

            logger.info('calling agent')
            # Final Answer (after tool use)
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                tools=tools,
                stream=True
            )
            logger.info('response received')
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # If NO tool needed (just chit-chat)
        else:
            st.markdown(response_msg.content)
            st.session_state.messages.append({"role": "assistant", "content": response_msg.content})