#import necessary packages
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import scipy.spatial.distance

# load data
data_path = os.path.join(DATA_DIR, "data")
df_eco_new = pd.read_parquet(data_path)

#Load the embedding model(here we use minilm)
model = SentenceTransformer('all-MiniLM-L6-v2')

logger = logging.getLogger("custom chat")

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

    logger.debug('Initiating bert search')
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
            "product_information": row.get("product information", None),
            "database_version": row.get("db", None)
        })
    return results