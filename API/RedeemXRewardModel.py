import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import ast
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import load_npz
import joblib
import pickle
from typing import List, Optional

app = FastAPI()

# Setting the working directory to the root of the project
base_dir = Path("/app")
data_dir = base_dir / "Data"


import pandas as pd
import numpy as np
from collections import defaultdict
pd.set_option('display.max_columns', None)

model = joblib.load(base_dir / 'redemptionxrewardals_model.joblib')


if hasattr(model, 'item_factors'):
    item_factors = model.item_factors
    if item_factors is None or item_factors.shape[0] == 0:
        raise ValueError("Item factors are not correctly loaded from the model.")
else:
    raise ValueError("Model does not have item_factors attribute.")


def load_user_profiles(cache_path):
    # Load the cached data if it exists
    user_profiles = joblib.load(cache_path)

    return user_profiles


user_profiles = load_user_profiles(data_dir / "cached_user_profiles.pkl")
with open(f"{base_dir}/redemption_user_item_matrix.pkl", 'rb') as file:
    user_item_matrix = pickle.load(file)


user_transactions = pd.read_excel(data_dir / "Transaction_User.xlsx")
new_user_transaction = user_transactions.drop(columns=['TrxId'])

deals_data = pd.read_excel(data_dir / "Rdepemtion_Cleaned_Deals.xlsx")
deals_data = deals_data.drop(columns=['Unnamed: 0.1'])

deals_profiles = pd.read_excel(data_dir / "Updated_Content_Profiles.xlsx")

deals_embeddings = pd.read_csv(data_dir / "Deals_Embeddings.csv")
deals_embeddings['ada_embedding'] = deals_embeddings['ada_embedding'].apply(ast.literal_eval)

new_user_transaction = new_user_transaction.merge(deals_data[['ContentId', 'Categories','Deal Type']], left_on='FK_ContentId', right_on='ContentId', how='left')
new_user_transaction = new_user_transaction.drop(columns=['ContentId'])

# Load MCC mapping data
mcc_mapping = pd.read_excel(data_dir / "MCC_Details.xlsx")
mcc_embeddings = pd.read_csv(data_dir / "Unique_MCCs_Embeddings.csv")

sparse_user_item = load_npz(base_dir / 'user_item_matrix.npz')

# Convert MCC in mcc_mapping to string
mcc_mapping['MCC'] = mcc_mapping['MCC'].astype(str)

print("all data loaded")
def is_numeric_column(col_name):
    try:
        int(col_name)
        return True
    except ValueError:
        return False


mcc_columns = [col for col in user_profiles.columns if is_numeric_column(col)]

mcc_amounts_cols = [col for col in user_profiles.columns.astype(str) if col.startswith('total_amount_mcc_')]

def calculate_user_interest_score(user_id):
    user_data = user_profiles[user_profiles['FK_BusinessUserId'] == user_id]
    
    total_frequencies = user_data[mcc_columns].sum(axis=1).values[0]
    total_paid_amount = user_data[mcc_amounts_cols].sum(axis=1).values[0]
    
    mcc_scores = {}
    if total_frequencies > 0 and total_paid_amount > 0:
        for mcc in mcc_columns:
            frequency = user_data[mcc].values[0]
            amount_col = f'total_amount_mcc_{mcc}'
            if amount_col in user_data.columns:
                amount = user_data[amount_col].values[0]
                if frequency > 0 and amount > 0:
                    score = (frequency / total_frequencies) * (amount / total_paid_amount)
                    mcc_scores[mcc] = score
    
    return mcc_scores


def convert_to_array(x):
    if isinstance(x, str):  # If x is a string, try to evaluate it
        try:
            return np.array(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            raise ValueError(f"Cannot convert to array: {x}")
    elif isinstance(x, (list, np.ndarray)):  # If x is already a list or array, convert to np.array
        return np.array(x)
    else:
        raise ValueError(f"Unexpected format: {x}")

def get_top_n_mccs(user_profiles , n=2):
    #Identifying MCC columns for amount and frequency in user_profiles
    mcc_amount_columns = [col for col in user_profiles.columns if str(col).startswith('total_amount_mcc_')]
    mcc_frequency_columns = [col for col in user_profiles.columns if str(col).isdigit()]

    # Normalizing the frequency and amount of MCCs
    mcc_amount_sums = user_profiles[mcc_amount_columns].sum()
    mcc_frequency_sums = user_profiles[mcc_frequency_columns].sum()

    normalized_amount = mcc_amount_sums / mcc_amount_sums.sum()
    normalized_frequency = mcc_frequency_sums / mcc_frequency_sums.sum()

    # Combining the metrics (we multiply with equal weight for simplicity)
    combined_score = 0.5 * normalized_amount.values + 0.5 * normalized_frequency.values 

    #Rank and selecting the top 10 MCCs
    combined_score_series = pd.Series(combined_score, index=mcc_frequency_columns)
    top_10_mccs = combined_score_series.sort_values(ascending=False).head(n)

    top_mccs_df = top_10_mccs.reset_index()
    top_mccs_df.columns = ['MCC', 'Score']
    top_mccs_df['MCC'] = top_mccs_df['MCC'].astype(str)
    top_mccs_with_details = pd.merge(top_mccs_df, mcc_mapping, on='MCC', how='left')
    top_mccs_with_details
    return top_mccs_with_details    

top_mccs = get_top_n_mccs(user_profiles , 2)

def recommend_based_on_profiles(user_id, deal_embeddings, deal_data, user_profiles,n_similar_items = 10 , isDf = False):
    # Implementing a recommendation strategy based on user profiles alone
    user_profile = user_profiles[user_profiles['FK_BusinessUserId'] == user_id]
    spender_category = user_profiles.loc[user_profiles['FK_BusinessUserId'] == user_id, 'spender_category'].values[0]
    print(f"Spender Category {spender_category}")
    
    #calculating mcc_scores
    mcc_scores = calculate_user_interest_score(user_id)

    # Merging the MCC scores with MCC mapping
    mcc_scores_df = pd.DataFrame.from_dict(mcc_scores, orient='index', columns=['Score'])
    mcc_scores_df.reset_index(inplace=True)
    mcc_scores_df.columns = ['MCC', 'Score']
    mcc_scores_df['MCC'] = mcc_scores_df['MCC'].astype(str)  # Convert MCC to string
    mcc_scores_df = mcc_scores_df.merge(mcc_mapping, on='MCC', how='left')
    mcc_scores_df = mcc_scores_df.sort_values(by='Score', ascending=False)

    # Creating labels combining MCC and description
    mcc_scores_df['Label'] = mcc_scores_df['MCC'] + ' - ' + mcc_scores_df['Detailed MCC']

    spender_ranges = {
        'low': ['Low-Budget Deal'],
        'medium': ['Medium Budget Deal','Low-Budget Deal'],
        'high': ['High-End Deal']
    }

    recommendations = []

    #user mccs embeddings
    mcc_embeddings['ada_embedding'] = mcc_embeddings['ada_embedding'].apply(convert_to_array)

    # Creating the embeddings dictionary
    mcc_embedding_dict = mcc_embeddings.set_index('MCC')['ada_embedding'].to_dict()

    # Filtering and collecting the embeddings that matches the user's MCC scores
    matched_embeddings = [mcc_embedding_dict[mcc] for mcc in mcc_scores.keys() if mcc in mcc_embedding_dict]


    user_embedding = np.mean(matched_embeddings, axis=0)


    # Getting top MCC scores
    mcc_scores_df = mcc_scores_df.sort_values(by='Score', ascending=False)

    user_mcc_scores = mcc_scores_df.set_index('Detailed MCC')['Score'].to_dict()

    
    for index, row in deal_data.iterrows():
        content_id = row['ContentId']
        item_mcc = row['Categories']
        deal_segment = deals_profiles.loc[deals_profiles['FK_ContentId'] == content_id, 'Deal Value Segment'].values[0]
        deal_embedding = np.array(deal_embeddings.loc[deal_embeddings['ContentId'] == content_id, 'ada_embedding'].values[0])
        score = cosine_similarity([user_embedding], [deal_embedding])[0][0]
        
        if item_mcc in top_mccs["Detailed MCC"].tolist():
            score *= 0.5

        # Adjusting the score based on MCC interest scores of the user
        if item_mcc in user_mcc_scores:
            score *= (1.2 + user_mcc_scores[item_mcc])
        
        
        # Adding weighted adjustment based on spender category
        if spender_category in spender_ranges and deal_segment in spender_ranges[spender_category]:
            score *= 1.5

        # recency = 1 / (1 + user_profile['recency'].values[0])
        # score *= (1 + 0.3 * recency)

        recommendations.append((content_id, score, row['Categories']))

    # Sorting recommendations by score
    sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    #Enforcing Diversity

    final_recommendations = []

    category_count = defaultdict(int)

    max_per_category = 2

    if(len(user_mcc_scores) < 4):
        max_per_category = 8


    for content_id, score, category in sorted_recommendations:
        if category_count[category] >= max_per_category:
            score *= 0.5  # Apply a penalty, could be adjusted dynamically
        if category_count[category] < max_per_category:
            final_recommendations.append((content_id, score))
            category_count[category] += 1

        if len(final_recommendations) >= n_similar_items:
            break
    
    if isDf:
        similar_item_ids = [item for item, score, category in final_recommendations]
        similar_item_scores = [score for item, score, category in final_recommendations]
        categories = [category for item, score, category in final_recommendations]
        recommendations_df = pd.DataFrame({
            'ContentId': similar_item_ids,
            'Score': similar_item_scores,
            'Category': categories
        })
        return recommendations_df
    else:    
        similar_item_ids = [item for item, score in final_recommendations]
        similar_item_scores = [score for item, score in final_recommendations]

    return similar_item_ids, similar_item_scores, spender_category


def collaborative_filtering_recommendations(user_id, user_item_matrix, model, deal_embeddings, deal_data, n_similar_items=10, new_deal_boost=0.0, popular_deal_penalty=0.0, category_penalty=0.0):
    user_index = list(user_item_matrix.index).index(user_id)
    user_interactions = sparse_user_item[user_index]
    
    
    interacted_items_indices = user_interactions.indices

    
    item_factors = model.item_factors

    
    filtered_embeddings = deal_embeddings[deal_embeddings['ContentId'].isin(user_item_matrix.columns)]


    deal_embeddings_array = np.array(filtered_embeddings['ada_embedding'].tolist())

    
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

    redeemed_counts = user_item_matrix.sum(axis=0)
    max_redeemed_count = redeemed_counts.max()
    for item in unique_similar_items.keys():
        if redeemed_counts[user_item_matrix.columns[item]] > 0:
            penalty = popular_deal_penalty * (redeemed_counts[user_item_matrix.columns[item]] / max_redeemed_count)
            unique_similar_items[item] -= penalty

    # Category-based penalty (to consider the category that the user have redeemed)
    user_redeemed_categories = deal_data[deal_data['ContentId'].isin(user_item_matrix.columns[interacted_items_indices])]['Categories'].unique()
    for item in unique_similar_items.keys():
        item_category = deal_data.loc[deal_data['ContentId'] == user_item_matrix.columns[item], 'Categories'].values[0]
        if item_category not in user_redeemed_categories:
            unique_similar_items[item] -= category_penalty

    similar_items_with_scores = sorted(unique_similar_items.items(), key=lambda x: x[1], reverse=True)

    # similar_item_indices = [item for item, score in similar_items_with_scores[:n_similar_items]]

    # similar_item_ids = [user_item_matrix.columns[item_id] for item_id in similar_item_indices]
    # similar_item_scores = [score for item, score in similar_items_with_scores[:n_similar_items]]

    recommended_deals_and_scores = [(user_item_matrix.columns[item_id], score) for item_id, score in similar_items_with_scores[:n_similar_items]]


    return recommended_deals_and_scores

def hybrid_recommendation_system(user_id, model,user_item_matrix, deal_embeddings, deal_data, user_profiles,n_recommendations=10):
    # Generating collaborative filtering recommendations
    cf_recommendations =  collaborative_filtering_recommendations(user_id, user_item_matrix, model,deal_embeddings,deal_data ,n_recommendations * 2 )

    
    # Generating content-based recommendations (users profiles + deals profiles)
    cb_recommendations, cb_scores , spender_category = recommend_based_on_profiles(user_id, deal_embeddings, deal_data, user_profiles, n_recommendations * 2)
    
    # Combining the recommendations
    combined_recommendations = {}
    
    
    for content_id, score in cf_recommendations:
        if content_id not in combined_recommendations:
            combined_recommendations[content_id] = 0.1 * score  # Lower weight for CF
    
    
    for content_id, score in zip(cb_recommendations, cb_scores):
        if content_id in combined_recommendations:
            combined_recommendations[content_id] += 0.9 * score  # adding Higher weight for CB (80 CF / 20 CB)
        else:
            combined_recommendations[content_id] = 0.9 * score
    
    # Sorting by combined scores
    sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # Enforcing category diversity
    final_recommendations = []
    category_count = {}
    
    for content_id, score in sorted_recommendations:
        final_recommendations.append((content_id, score))
        
        if len(final_recommendations) >= n_recommendations:
            break
    
    return final_recommendations , spender_category


# FastAPI input model for the request
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

class DealResponse(BaseModel):
    ContentId: int
    Score: float
    DealName: str
    Category: str
    DealType: str
    Points : int
class FullRecommendationResponse(BaseModel):
    UserSpenderCategory: Optional[str]
    Deals: List[DealResponse]


@app.post("/recommend/", response_model=FullRecommendationResponse)
async def recommend(request: RecommendationRequest):
    user_id = request.user_id
    n_recommendations = request.n_recommendations

    # Checking if the user_id is in the user_item_matrix (CF)
    user_in_item_matrix = user_id in user_item_matrix.index

    # Checking if the user_id is in the user_profiles (CB)
    user_in_profiles = user_id in user_profiles['FK_BusinessUserId'].values

    if user_in_item_matrix and user_in_profiles:
        print("Hybrid Recommendation")
        # Calling hybrid_recommendation_system if user_id is in both
        hybrid_recommended_deals, spender_category = hybrid_recommendation_system(
            user_id, model, user_item_matrix, deals_embeddings, deals_data, user_profiles, n_recommendations
        )
        recommended_deals = hybrid_recommended_deals

    elif user_in_item_matrix:
        print("Collaborative Filtering Only")
        # Only call collaborative_filtering_recommendations if the user is in the user_item_matrix
        recommended_deals = collaborative_filtering_recommendations(
            user_id, user_item_matrix, model, deals_embeddings, deals_data, n_recommendations
        )
        spender_category = None  # Not applicable in CF-only mode

    elif user_in_profiles:
        print("Content-Based Filtering Only")
        # Only call recommend_based_on_profiles if the user is in the user_profiles
        cb_recommendations, cb_scores, spender_category = recommend_based_on_profiles(
            user_id, deals_embeddings, deals_data, user_profiles, n_similar_items=n_recommendations
        )
        recommended_deals = list(zip(cb_recommendations, cb_scores))

    else:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Build the response
    deals_list = [
        {
            'ContentId': content_id,
            'Score': score,
            'DealName': deals_data.loc[deals_data['ContentId'] == content_id, 'Title'].values[0],
            'Category': deals_data.loc[deals_data['ContentId'] == content_id, 'Categories'].values[0],
            'DealType': deals_data.loc[deals_data['ContentId'] == content_id, 'Deal Type'].values[0],
            'Points': deals_data.loc[deals_data['ContentId'] == content_id, 'Points'].values[0],
        }
        for content_id, score in recommended_deals
    ]

    return FullRecommendationResponse(
        UserSpenderCategory=spender_category,
        Deals=deals_list
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)