import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df = pd.read_csv('cleaned_booking_reviews.csv')

# Aggregate duplicate entries by taking the average rating
df_agg = df.groupby(['reviewed_by', 'hotel_name'])['rating'].mean().reset_index()

# Create the User-Item matrix
user_item_matrix = df_agg.pivot(index='reviewed_by', columns='hotel_name', values='rating')
user_item_matrix = user_item_matrix.fillna(0)  # Fill missing values with 0

# Initialize the k-NN model and fit the user-item matrix
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='auto')
knn_model.fit(user_item_matrix)

# Function to get user-based recommendations
def get_user_based_recommendations(user_id):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found in the user-item matrix.")
        return []

    # Get similar users
    distances, indices = knn_model.kneighbors(user_item_matrix.loc[user_id].values.reshape(1, -1))
    print(f"Distances: {distances}")  # Debugging: print distances
    print(f"Indices: {indices}")  # Debugging: print indices

    # Get recommendations for hotels based on similar users
    similar_users = indices[0][1:6]  # Exclude the first index (the user itself)
    similar_users_ratings = user_item_matrix.iloc[similar_users]
    mean_ratings = similar_users_ratings.mean(axis=0)
    
    # Sort hotels by the highest mean ratings and return top 5 recommendations
    recommended_hotels = mean_ratings.sort_values(ascending=False).head(5)
    print(f"Recommended Hotels: {recommended_hotels}")  # Debugging: print recommended hotels
    return recommended_hotels.index.tolist()

# To get a valid user ID from the user-item matrix, print all user IDs in the matrix
print("User IDs in the user-item matrix:")
print(user_item_matrix.index)

# Example of using a valid user ID from the printed list
user_id = "7sain_89"  # Replace this with a valid user ID from the above printout
recommendations = get_user_based_recommendations(user_id)
print("Recommendations:", recommendations)
