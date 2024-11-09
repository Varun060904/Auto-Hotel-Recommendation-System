import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to calculate Precision, Recall, and F1-Score at K
def evaluate_precision_recall_f1(y_true, y_pred, k=5):
    """
    Evaluate the precision, recall, and F1-score of the top K recommendations.

    Args:
    y_true (list): List of actual hotels that the user interacted with.
    y_pred (list): List of recommended hotels.
    k (int): Number of top recommendations to consider.

    Returns:
    tuple: Precision, Recall, F1-score
    """
    # Ensure y_pred is a list of hotel names (in case it's a pandas Series or DataFrame index)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.index.tolist()  # Convert DataFrame/Series index to list if needed
    
    # Only consider the top K recommendations
    top_k_predictions = y_pred[:k]

    # Debugging: Print the actual ground truth and predicted recommendations
    print(f"Ground truth (y_true): {y_true}")
    print(f"Top K recommendations (y_pred): {top_k_predictions}")
    
    # Calculate True Positives (number of hotels in both y_true and y_pred)
    true_positive = sum(1 for hotel in top_k_predictions if hotel in y_true)
    
    # Precision: TP / K
    precision = true_positive / k if k > 0 else 0
    
    # Recall: TP / number of relevant hotels (length of y_true)
    recall = true_positive / len(y_true) if len(y_true) > 0 else 0
    
    # F1-Score: Harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Simulate hotel recommendation data (Replace this with your actual recommendation models)
collab_recommendations = ['Novotel Brussels City Centre', 'Chimay B & B', 'En Aqualye', 'Ter Ename n°2', "t Eenvoud"]
content_recommendations = ['ABC Hotel', 'Hotel Europe', 'Mercure Hotel Brussels Centre Midi', 'Novotel Gent Centrum', 'The best known village of Belgium']
hybrid_recommendations = ['Novotel Brussels City Centre', 'Hotel Dolce La Hulpe Brussels', 'ABC Hotel', 'Hotel Europe', 'Mercure Hotel Brussels Centre Midi']

# Simulated ground truth (replace with actual user data)
# Assume 'Syed' has interacted with these hotels
y_true = ['Novotel Brussels City Centre', 'Hotel Dolce La Hulpe Brussels', 'ABC Hotel', 'Hotel Europe']

# Evaluate Collaborative Filtering
print("Evaluating Collaborative Filtering...")
collab_precision, collab_recall, collab_f1 = evaluate_precision_recall_f1(y_true, collab_recommendations, k=5)
print(f"Collaborative Filtering Metrics: Precision={collab_precision}, Recall={collab_recall}, F1={collab_f1}")

# Evaluate Content-Based Filtering
print("Evaluating Content-Based Filtering...")
content_precision, content_recall, content_f1 = evaluate_precision_recall_f1(y_true, content_recommendations, k=5)
print(f"Content-Based Filtering Metrics: Precision={content_precision}, Recall={content_recall}, F1={content_f1}")

# Evaluate Hybrid Filtering
print("Evaluating Hybrid Filtering...")
hybrid_precision, hybrid_recall, hybrid_f1 = evaluate_precision_recall_f1(y_true, hybrid_recommendations, k=5)
print(f"Hybrid Filtering Metrics: Precision={hybrid_precision}, Recall={hybrid_recall}, F1={hybrid_f1}")

# Summary of Evaluation Metrics
print("\nEvaluation Metrics:")
print(f"Collaborative: Precision = {collab_precision}, Recall = {collab_recall}, F1-Score = {collab_f1}")
print(f"Content-Based: Precision = {content_precision}, Recall = {content_recall}, F1-Score = {content_f1}")
print(f"Hybrid: Precision = {hybrid_precision}, Recall = {hybrid_recall}, F1-Score = {hybrid_f1}")

# Optionally, you can add visualizations using matplotlib
import matplotlib.pyplot as plt

def plot_metrics(metrics, model_names, metric_name):
    """
    Plot Precision, Recall, or F1-Score for each model.
    
    Args:
    metrics (list): List of metric values.
    model_names (list): List of model names.
    metric_name (str): Metric name (Precision, Recall, or F1).
    """
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, metrics, color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} of Different Models')
    plt.show()

# Plot Precision, Recall, and F1 for each model
plot_metrics([collab_precision, content_precision, hybrid_precision], ['Collaborative', 'Content-Based', 'Hybrid'], 'Precision')
plot_metrics([collab_recall, content_recall, hybrid_recall], ['Collaborative', 'Content-Based', 'Hybrid'], 'Recall')
plot_metrics([collab_f1, content_f1, hybrid_f1], ['Collaborative', 'Content-Based', 'Hybrid'], 'F1-Score')
