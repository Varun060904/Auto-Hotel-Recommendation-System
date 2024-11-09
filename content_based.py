import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to fill missing values and preprocess data
def preprocess_data(df):
    df['review_text'].fillna('', inplace=True)  # Fill missing reviews with empty string
    df['tags'].fillna('No Tags', inplace=True)  # Fill missing tags with 'No Tags'
    return df

# Function to get content-based recommendations for a user
def get_content_based_recommendations(user_id, df):
    # Ensure the user ID exists in the user-item matrix
    if user_id not in df['reviewed_by'].values:
        print(f"User ID {user_id} not found in the user-item matrix.")
        print(f"Available user IDs: {df['reviewed_by'].unique()[:10]}")  # Display the first 10 available user IDs
        return []

    # Filter the dataset for the user's reviews
    user_reviews = df[df['reviewed_by'] == user_id]
    
    # Convert hotel names into a list
    user_hotels = user_reviews['hotel_name'].tolist()
    
    # Apply TF-IDF vectorization on text columns (review_text, tags, hotel_name)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Combine relevant text columns to create a feature set
    df['combined_text'] = df['hotel_name'] + " " + df['tags'] + " " + df['review_text']
    df_features = tfidf_vectorizer.fit_transform(df['combined_text'])

    # Create a nearest neighbors model
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    model.fit(df_features)

    recommended_hotels = []
    for hotel in user_hotels:
        try:
            idx = df[df['hotel_name'] == hotel].index[0]
            distances, indices = model.kneighbors(df_features[idx].reshape(1, -1), n_neighbors=5)
            for i in indices[0]:
                recommended_hotels.append(df.iloc[i]['hotel_name'])
        except IndexError:
            continue

    recommended_hotels = list(set(recommended_hotels))  # Remove duplicates
    print(f"Top 5 content-based recommended hotels for '{user_id}':")
    return recommended_hotels[:5]  # Return top 5 recommendations

# Example usage
if __name__ == "__main__":
    # Load your dataset (replace with the correct path)
    df = pd.read_csv('cleaned_booking_reviews.csv')

    # Preprocess the data
    df = preprocess_data(df)

    # Set user ID (use a valid ID from your dataset)
    user_id = 'Aaro'  # Replace with an actual user ID from your dataset
    recommended_hotels = get_content_based_recommendations(user_id, df)

    print("Recommended Hotels:", recommended_hotels)
