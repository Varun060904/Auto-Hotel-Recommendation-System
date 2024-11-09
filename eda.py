import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (after preprocessing)
df = pd.read_csv('cleaned_booking_reviews.csv')  # replace with your dataset path

# Set plot style for better visuals
sns.set(style="whitegrid")

# 1. Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['avg_rating'], kde=True, bins=20, color='blue')
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# 2. Check Missing Data
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# 3. Basic Statistical Summary (numeric columns)
print("Basic Statistical Summary:")
print(df.describe())

# 4. Review Text Length Distribution
df['review_text_length'] = df['review_text'].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 6))
sns.histplot(df['review_text_length'], kde=True, color='green', bins=20)
plt.title('Distribution of Review Text Length')
plt.xlabel('Review Text Length')
plt.ylabel('Frequency')
plt.show()

# 5. Most Common Hotel Names
top_hotels = df['hotel_name'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_hotels.index, y=top_hotels.values, palette='Blues_d')
plt.title('Top 10 Most Common Hotel Names')
plt.xlabel('Hotel Name')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.show()

# 6. Nationality Distribution
top_nationalities = df['nationality'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_nationalities.index, y=top_nationalities.values, palette='viridis')
plt.title('Top 10 Nationalities of Reviewers')
plt.xlabel('Nationality')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45, ha='right')
plt.show()

# 7. Tag Frequency Analysis (Top 10 Most Frequent Tags)
# First, we need to handle the 'tags' column, which may have lists in some rows.
df['tags_count'] = df['tags'].apply(lambda x: len(eval(str(x))) if pd.notnull(x) else 0)

# Top 10 most frequent tag counts
top_tags = df['tags_count'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_tags.index, y=top_tags.values, palette='coolwarm')
plt.title('Top 10 Tag Counts')
plt.xlabel('Number of Tags')
plt.ylabel('Frequency')
plt.show()

# 8. Correlation Heatmap (for numerical columns)
correlation_matrix = df[['avg_rating', 'rating', 'review_text_length']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Ratings and Review Length')
plt.show()
