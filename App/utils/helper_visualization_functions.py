import matplotlib.dates as mdates
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging
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


def cluster_visualization_of_time_series(results_with_cluster_id=None, cluster_dir_path=None):
    # Ensure that the cluster directory exists
    if not os.path.exists(cluster_dir_path):
        os.makedirs(cluster_dir_path)

    # Group by cluster and visualize each cluster separately
    for cluster_id, cluster_group in results_with_cluster_id.groupby('cluster'):
        print(f"Cluster ID: {cluster_id}")
        print(f"Cluster id : {cluster_group['cluster'].unique()}")
        
        print(f"Number of tokens in this cluster: {len(cluster_group['base_currency'].unique())}")

        plt.figure(figsize=(12, 6))
        for token_id, token_group in cluster_group.groupby('base_currency'):
            plt.plot(token_group['timestamp_utc'], token_group['open'], label=f'Token {token_id}')
        
        # Add labels, title, and formatting for the plot
        plt.xlabel('Time (UTC)')
        plt.ylabel('Open Price')
        plt.title(f'Open Price Over Time for Tokens in Cluster {cluster_id}')
        plt.xticks(rotation=45)
        plt.legend(loc='best')  # Optional: Show a legend for token IDs
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        plot_file_path = os.path.join(cluster_dir_path, f"cluster_{cluster_id}.png")
        print(f"file saved at path : {plot_file_path}")
        plt.savefig(plot_file_path)

        # Close the plot before showing it to avoid overlapping
        plt.close()

        # Optionally show the plot
        plt.show()


def generate_wordclouds_for_clusters(token_names_df=None, directory_names=None, cluster_label_name=None):
    # Group by the 'token_cluster' column
    grouped = token_names_df.groupby(cluster_label_name)

    for cluster, group in grouped:
        # try:
        # Skip invalid clusters
        if cluster < 0:
            logging.warning(f"Skipping invalid cluster: {cluster}")
            continue

        number_of_tokens = group.shape[0]
        logging.info(f"Generating word cloud for Cluster {cluster} with {number_of_tokens} tokens.")

        # Ensure there are tokens for this cluster
        if number_of_tokens == 0:
            logging.warning(f"No tokens found for cluster {cluster}. Skipping.")
            continue

        # Generate word cloud strings
        cluster_words_display_name = ' '.join(group['display_name'].astype(str))
        cluster_words_full_name = ' '.join(group['full_name'].astype(str).fillna(''))
        cluster_words_full_name_id = ' '.join(list(group['id'].astype(str)))

        # Generate word clouds
        wordcloud_display_name = WordCloud(width=800, height=400, background_color='white',
                                           max_words=100, colormap='viridis').generate(cluster_words_display_name)
        wordcloud_full_name = WordCloud(width=800, height=400, background_color='white',
                                        max_words=100, colormap='plasma').generate(cluster_words_full_name)

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(wordcloud_display_name, interpolation='bilinear')
        ax[0].axis('off')
        ax[0].set_title(
            f'Cluster {cluster} - Display Name, Number of Tokens: {number_of_tokens} : Token ids :{cluster_words_full_name_id}')
        ax[1].imshow(wordcloud_full_name, interpolation='bilinear')
        ax[1].axis('off')
        ax[1].set_title(f'Cluster {cluster} - Full Name, Number of Tokens: {number_of_tokens}')

        output_path = os.path.join(directory_names['cluster_token_name_dir'],
                                   f"wordcloud_cluster_{cluster}.png")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        # plt.show()
        logging.info(f"Word cloud for Cluster {cluster} saved at: {output_path}")
