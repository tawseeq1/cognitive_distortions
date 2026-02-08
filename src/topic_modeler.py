from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import numpy as np
from .config import Config

class TopicModeler:
    def __init__(self, model_name=Config.MODEL_NAME):
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, sentences):
        """
        Returns embeddings for a list of sentences.
        """
        print(f"Generating embeddings for {len(sentences)} sentences...")
        # Encode in batches to avoid OOM, though sentence-transformers handles it well usually
        return self.model.encode(sentences, show_progress_bar=True)

    def find_optimal_clusters(self, embeddings, k_min=10, k_max=100, k_step=10):
        """
        Tests multiple K values and returns the best model based on Davies-Bouldin score.
        """
        best_score = float('inf')
        best_k = k_min
        results = []

        print("Finding optimal clusters...")
        for k in range(k_min, k_max + 1, k_step):
            if k >= len(embeddings): # Cannot have more clusters than samples
                break
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5) # n_init=5 for speed
            labels = kmeans.fit_predict(embeddings)
            score = davies_bouldin_score(embeddings, labels)
            
            print(f"K={k}, DB Score={score:.4f}")
            results.append({'k': k, 'score': score})
            
            if score < best_score:
                best_score = score
                best_k = k
        
        print(f"Best K found: {best_k} with DB Score: {best_score:.4f}")
        return best_k, results

    def run_clustering(self, sentences_df, distortion_name):
        """
        Full pipeline for a specific distortion subset.
        """
        subset = sentences_df[sentences_df[distortion_name] == True].copy()
        
        if len(subset) < 20:
             print(f"Not enough data for clustering {distortion_name} (n={len(subset)})")
             return None

        embeddings = self.generate_embeddings(subset['sentence'].tolist())
        
        # Find best K
        best_k, _ = self.find_optimal_clusters(embeddings, 
                                               k_min=Config.CLUSTERS_K_MIN, 
                                               k_max=Config.CLUSTERS_K_MAX, 
                                               k_step=Config.CLUSTERS_K_STEP)
        
        # Final Cluster
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        subset['cluster'] = labels
        return subset
