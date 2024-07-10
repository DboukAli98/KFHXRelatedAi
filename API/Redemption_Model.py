from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
from pathlib import Path

app = FastAPI()

# Defining the project directory and source data path
project_dir = Path("C:/Users/adbou/source/repos/KFHXRelatedAi/")
source_data_dir = project_dir / "Api"

# Loading the model and data
model = joblib.load(source_data_dir / 'als-model.joblib')


if hasattr(model, 'item_factors'):
    item_factors = model.item_factors
    if item_factors is None or item_factors.shape[0] == 0:
        raise ValueError("Item factors are not correctly loaded from the model.")
else:
    raise ValueError("Model does not have item_factors attribute.")

sparse_user_item = load_npz(source_data_dir / 'user_item_matrix.npz')
user_item_matrix_df = pd.read_pickle(source_data_dir / 'user_item_matrix_df.pkl')
deals_data = pd.read_csv(source_data_dir / 'deals_data.csv')
deals_embeddings = pd.read_csv(source_data_dir / 'deals_embeddings.csv')
deals_embeddings['ada_embedding'] = deals_embeddings['ada_embedding'].apply(ast.literal_eval)

class RecommendationRequest(BaseModel):
    user_id: int
    n_similar_items: int = 10

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        similar_item_ids, similar_item_scores = recommend_items(
            request.user_id, user_item_matrix_df, sparse_user_item, model, deals_embeddings, deals_data, request.n_similar_items
        )
        # Converting numpy.int64 to native int
        similar_item_ids = [int(item_id) for item_id in similar_item_ids]
        similar_item_scores = [float(score) for score in similar_item_scores]
        return {"user_id": request.user_id, "recommended_items": similar_item_ids, "scores": similar_item_scores}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def recommend_items(user_id, user_item_matrix_df, sparse_user_item, model, deal_embeddings, deal_data, n_similar_items=10, new_deal_boost=0.1, popular_deal_penalty=0.2, category_penalty=0.4):
    if user_id not in user_item_matrix_df.index:
        raise ValueError(f"User ID {user_id} not found.")
    
    user_index = user_item_matrix_df.index.get_loc(user_id)
    user_interactions = sparse_user_item[user_index]

    interacted_items_indices = user_interactions.indices

    item_factors = model.item_factors

    filtered_embeddings = deal_embeddings[deal_embeddings['ContentId'].isin(user_item_matrix_df.columns)]

    deal_embeddings_array = np.array(filtered_embeddings['ada_embedding'].tolist())

    if deal_embeddings_array is None or deal_embeddings_array.shape[0] == 0:
        raise ValueError("Deal embeddings array is empty or not correctly loaded.")

    similarity_matrix_factors = cosine_similarity(item_factors)

    similarity_matrix_embeddings = cosine_similarity(deal_embeddings_array)

    min_shape = min(similarity_matrix_factors.shape[0], similarity_matrix_embeddings.shape[0])
    similarity_matrix_factors = similarity_matrix_factors[:min_shape, :min_shape]
    similarity_matrix_embeddings = similarity_matrix_embeddings[:min_shape, :min_shape]

    combined_similarity_matrix = (similarity_matrix_factors + similarity_matrix_embeddings) / 2.0

    unique_similar_items = {}

    for item_index in interacted_items_indices:
        similar_items = combined_similarity_matrix[item_index].argsort()[::-1][1:n_similar_items+1]
        for similar_item in similar_items:
            if similar_item not in interacted_items_indices:
                score = combined_similarity_matrix[item_index][similar_item]
                if similar_item in unique_similar_items:
                    unique_similar_items[similar_item] = max(unique_similar_items[similar_item], score)
                else:
                    unique_similar_items[similar_item] = score

    all_deal_indices = set(range(combined_similarity_matrix.shape[0]))
    non_redeemed_deals = all_deal_indices - set(interacted_items_indices)
    for deal in non_redeemed_deals:
        if deal in unique_similar_items:
            unique_similar_items[deal] = max(unique_similar_items[deal], new_deal_boost)
        else:
            unique_similar_items[deal] = new_deal_boost

    redeemed_counts = user_item_matrix_df.sum(axis=0)
    max_redeemed_count = redeemed_counts.max()
    for item in unique_similar_items.keys():
        if redeemed_counts[user_item_matrix_df.columns[item]] > 0:
            penalty = popular_deal_penalty * (redeemed_counts[user_item_matrix_df.columns[item]] / max_redeemed_count)
            unique_similar_items[item] -= penalty

    user_redeemed_categories = deal_data[deal_data['ContentId'].isin(user_item_matrix_df.columns[interacted_items_indices])]['Categories'].unique()
    for item in unique_similar_items.keys():
        item_category = deal_data.loc[deal_data['ContentId'] == user_item_matrix_df.columns[item], 'Categories'].values[0]
        if item_category not in user_redeemed_categories:
            unique_similar_items[item] -= category_penalty

    similar_items_with_scores = sorted(unique_similar_items.items(), key=lambda x: x[1], reverse=True)

    similar_item_indices = [item for item, score in similar_items_with_scores[:n_similar_items]]

    similar_item_ids = [user_item_matrix_df.columns[item_id] for item_id in similar_item_indices]
    similar_item_scores = [score for item, score in similar_items_with_scores[:n_similar_items]]

    return similar_item_ids, similar_item_scores

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
