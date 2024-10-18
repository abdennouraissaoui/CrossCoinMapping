
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

# Function to plot and save the time series comparison
def plot_and_save(ts1, ts2, token1, token2, similarity_score, saving_file_path):
    plt.figure(figsize=(18, 6))
    
    # Plot time series
    plt.plot(ts1.index, ts1, label=token1, color='blue')
    plt.plot(ts2.index, ts2, label=token2, color='orange')
    
    # Set title and labels
    plt.title(f"DTW Distance between {token1} and {token2}: metrics : {similarity_score}", fontsize=14)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    
    # Format x-axis to show date labels clearly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate and align the date labels
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(saving_file_path)
    plt.close()  # Close the figure

def calculate_cosine_similarity_char(str1, str2, ngram_range=(2, 3)):
    
    # Create a TF-IDF vectorizer that works with character n-grams
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    
    # Vectorize the input strings using character-level n-grams
    vectors = vectorizer.fit_transform([str1, str2])
    
    # Calculate cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectors[0], vectors[1])
    # print(f"difference between : {str1} and {str2} is : {cosine_sim[0][0]}")
    return cosine_sim[0][0]