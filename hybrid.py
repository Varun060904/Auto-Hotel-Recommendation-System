import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the preprocessed data
df = pd.read_csv("cleaned_booking_reviews.csv")

# Collaborative Filtering Model
user_item_matrix = df.pivot_table(index='reviewed_by', columns='hotel_name', values='rating').fillna(0)
user_knn = NearestNeighbors(metric='cosine', algorithm='brute')
user_knn.fit(user_item_matrix)

def get_collab_recommendations(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found in dataset.")
        return None
    
    distances, indices = user_knn.kneighbors(user_item_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=6)
    similar_users = indices.flatten()[1:]
    
    hotel_scores = user_item_matrix.iloc[similar_users].mean(axis=0)
    return hotel_scores.nlargest(num_recommendations)

# Content-Based Model
df['combined_text'] = df['review_text'] + ' ' + df['tags']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_text'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(hotel_name, num_recommendations=5):
    if hotel_name not in df['hotel_name'].values:
        print(f"Hotel '{hotel_name}' not found in dataset.")
        return None
    
    hotel_idx = df[df['hotel_name'] == hotel_name].index[0]
    similarity_scores = list(enumerate(content_sim[hotel_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_hotels = [df.iloc[i[0]]['hotel_name'] for i in similarity_scores[1:num_recommendations + 1]]
    return top_hotels

# Hybrid Recommendation
def hybrid_recommendation(user_id, hotel_name, num_recommendations=5, weight_collab=0.5, weight_content=0.5):
    collab_scores = get_collab_recommendations(user_id, num_recommendations * 2)
    if collab_scores is None:
        return []
    
    content_recommendations = get_content_recommendations(hotel_name, num_recommendations * 2)
    if content_recommendations is None:
        return []
    
    hybrid_scores = {}
    for hotel, score in collab_scores.items():
        hybrid_scores[hotel] = weight_collab * score
    
    for hotel in content_recommendations:
        if hotel in hybrid_scores:
            hybrid_scores[hotel] += weight_content * 1
        else:
            hybrid_scores[hotel] = weight_content * 1
    
    sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:num_recommendations]

# Test with a valid user ID and hotel name
user_id = user_item_matrix.index[0]  # Use an actual user ID from user_item_matrix
hotel_name = "Novotel Brussels City Centre"  # Ensure this exists in the dataset

recommendations = hybrid_recommendation(user_id, hotel_name)
print("Hybrid Recommendations:")
for hotel, score in recommendations:
    print(f"{hotel}: {score:.2f}")
