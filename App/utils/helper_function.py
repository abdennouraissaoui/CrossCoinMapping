
# Function to calculate DTW distance
def calculate_dtw_distance(ts1, ts2):
    ts1 = np.array(ts1).reshape(-1, 1)
    ts2 = np.array(ts2).reshape(-1, 1)
    dtw_distance, _, _, _ = accelerated_dtw(ts1, ts2, dist='euclidean')
    return float(np.round(dtw_distance, 2) )

# Function to calculate error metrics
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