from utils.helper_clustering_functions import KMeanClustering ,kmeans_with_smape_ts ,kmeans_with_min_distance
from utils.helper_similarity_metrics import calculate_dtw_distance , calculate_error_metrics ,calculate_cosine_similarity_char ,CoinCrossMappingSimilarity ,smape,smape_distance_metric
from utils.helper_visualization_functions import plot_and_save , cluster_visualization_of_time_series
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import csv
import os
import glob
import tqdm
import gc
import logging
from dotenv import load_dotenv
load_dotenv()

directory_names = {
    "preprocessed_data_dir_name": "ProcessedData",
    "visualization_data_dir_name": "VisualizationData",
    "testing_garbage_dir_name": "TestingGarbage",
    "ResultsDirectory": "SimilarityResults",
    "cluster_dir_path": "ClusterResultsVisualization",
    "cluster_token_name_dir": "ClusterBaseOnTokenNameVisualization"
}
for key, value in directory_names.items():
    print(f"Creating Directories : {value}")
    os.makedirs(directory_names[key], exist_ok=True)

files_path = {
    'raw_price_data': os.path.join("Datasets", "raw_datasets", "prices.csv"),
    'raw_token_names': os.path.join("Datasets", "raw_datasets", "token_names.csv"),
    "token_names": os.path.join("TestingGarbage", "token_names.csv"),
    "similarity_results_file_path": os.path.join("SimilarityResults", "similarity_results_version_0.1.csv")
}