import pandas as pd

# Load the dataset
df = pd.read_csv('sampled_booking_reviews.csv')

# Check for missing data
print("Missing Data:")
print(df.isnull().sum())

# Handle missing values for numerical columns by filling with median
df['avg_rating'].fillna(df['avg_rating'].median(), inplace=True)
df['rating'].fillna(df['rating'].median(), inplace=True)

# Handle missing values for text columns (e.g., 'review_text' and 'tags')
df['review_text'].fillna('', inplace=True)  # Empty string for review text
df['tags'].fillna('No Tags', inplace=True)  # 'No Tags' for missing tags

# Drop rows where key information is missing (critical columns like 'hotel_name', 'review_title', or 'rating')
df.dropna(subset=['hotel_name', 'review_title', 'rating'], inplace=True)

# Double-check if there are still missing values
print("\nAfter Handling Missing Data:")
print(df.isnull().sum())

# You can also print out the first few rows to see how the data looks now
print("\nFirst few rows after preprocessing:")
print(df.head())

# Save the cleaned dataset for further use
df.to_csv('cleaned_booking_reviews.csv', index=False)
