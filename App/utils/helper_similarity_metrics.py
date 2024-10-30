import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dtw import accelerated_dtw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .helper_visualization_functions import plot_and_save
import os
import csv
import gc
from tqdm import tqdm
import glob

def calculate_dtw_distance(ts1, ts2):
    ts1 = np.array(ts1).reshape(-1, 1)
    ts2 = np.array(ts2).reshape(-1, 1)
    dtw_distance, _, _, _ = accelerated_dtw(ts1, ts2, dist='euclidean')
    return float(np.round(dtw_distance, 2) )

def calculate_error_metrics(ts1, ts2):
    mae = float(mean_absolute_error(ts1, ts2))
    rmse = float(np.sqrt(mean_squared_error(ts1, ts2)))
    mape = float(np.mean(np.abs((ts1 - ts2) / ts1)) * 100)  # MAPE
    smape = float(100 * np.mean(2 * np.abs(ts1 - ts2) / (np.abs(ts1) + np.abs(ts2))))  # SMAPE
    
    return round(mae, 2), round(rmse, 2), round(mape, 2), round(smape, 2)

def calculate_cosine_similarity_char(str1, str2, ngram_range=(2, 3)):
    
    # Create a TF-IDF vectorizer that works with character n-grams
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    
    # Vectorize the input strings using character-level n-grams
    vectors = vectorizer.fit_transform([str1, str2])
    
    # Calculate cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectors[0], vectors[1])
    # print(f"difference between : {str1} and {str2} is : {cosine_sim[0][0]}")
    return cosine_sim[0][0]


def CoinCrossMappingSimilarity(results_with_cluster_id=None, price_pivot_df=None, files_path=None, directory_names=None):
    results_file = files_path['similarity_results_file_path']

    if not os.path.exists(results_file):
        with open(results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Token1', 'Token2', 'DTW_Distance', 'MAE', 'RMSE', 'MAPE', 'SMAPE', "cosine_sim"])

    # Iterate through each cluster
    for cluster_id, cluster_group in results_with_cluster_id.groupby('cluster'):
        print(f"Processing Cluster ID: {cluster_id}")
        tokens = list(cluster_group['token_id'].unique())
        print(f"Number of tokens in this cluster: {len(tokens)}")

        # Loop through each pair of tokens (avoid redundant pairs with i < j)
        for i, token1 in tqdm(enumerate(tokens), total=len(tokens), desc=f'Cluster {cluster_id}'):
            token1 = str(token1)
            for j, token2 in enumerate(tokens):
                token2 = str(token2)
                if i >= j:  # Skip redundant calculations
                    continue

                try:
                    # Make sure to convert paths and token IDs to strings
                    token_folder_name = os.path.join(directory_names['visualization_data_dir_name'], token1)
                    os.makedirs(token_folder_name, exist_ok=True)

                    plot_filename = f"{token1}_vs_{token2}_dtw.png"
                    saving_file_path = os.path.join(token_folder_name, plot_filename)

                    # Check if the plot already exists
                    if os.path.exists(saving_file_path):
                        continue

                    print(f"Token1: {token1} and Token2: {token2}")

                    # Drop missing values and find common indices
                    ts1 = price_pivot_df[token1].dropna()
                    ts2 = price_pivot_df[token2].dropna()
                    common_index = ts1.index.intersection(ts2.index)
                    ts1_common = ts1.loc[common_index]
                    ts2_common = ts2.loc[common_index]

                    # Proceed if common data points exist
                    if len(ts1_common) > 0 and len(ts2_common) > 0:
                        # Calculate similarity metrics
                        dtw_distance = calculate_dtw_distance(ts1_common, ts2_common)
                        cosine_sim = calculate_cosine_similarity_char(token1, token2)

                        # Calculate error metrics
                        mae, rmse, mape, smape = calculate_error_metrics(ts1_common, ts2_common)
                        similarity_score = {
                            "mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                            "smape": smape,
                            "dtw": dtw_distance,
                            "cosine_sim": cosine_sim
                        }

                        # Save plot only if SMAPE is above 15
                        if smape < 15:
                            plot_and_save(ts1=ts1_common, 
                                          ts2=ts2_common, 
                                          token1=token1, 
                                          token2=token2, 
                                          similarity_score=similarity_score, 
                                          saving_file_path=saving_file_path)

                        # Save the results in a CSV file
                        print(f"SMAPE: {smape}")

                        with open(results_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([token1, token2, dtw_distance, mae, rmse, mape, smape, cosine_sim])

                        # Free memory if necessary
                        del ts1_common, ts2_common
                        gc.collect()

                except Exception as e:
                    print(f"Error processing tokens {token1} and {token2}: {e}")
                    raise e
                    continue

    gc.collect()
    total_detected_similar_tokens = len(glob.glob(f"{directory_names['visualization_data_dir_name']}/*/*.png"))
    return total_detected_similar_tokens

def smape(y_true, y_pred):
    """Calculate SMAPE between two time series."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


# Custom distance metric (SMAPE) for DBSCAN
def smape_distance_metric(ts1, ts2):
    return smape(ts1, ts2)