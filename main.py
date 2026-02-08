import argparse
import os
import sys
import pandas as pd
from src.config import Config
from src.data_loader import DataLoader
from src.distortion_detector import DistortionDetector
from src.visualizer import Visualizer
from src.topic_modeler import TopicModeler

def main():
    parser = argparse.ArgumentParser(description="Cognitive Distortion Analysis Pipeline")
    
    parser.add_argument('--rows', type=int, default=None, help="Number of rows to process (for testing)")
    parser.add_argument('--posts_path', type=str, default=os.path.join(Config.RAW_DATA_DIR, Config.POSTS_FILENAME), help="Path to posts CSV")
    parser.add_argument('--comments_path', type=str, default=os.path.join(Config.RAW_DATA_DIR, Config.COMMENTS_FILENAME), help="Path to comments CSV")
    parser.add_argument('--mode', type=str, choices=['all', 'topic_model'], default='all', help="Analysis mode")
    
    args = parser.parse_args()
    
    # 1. Setup
    print("Initializing components...")
    Config.ensure_directories()
    loader = DataLoader()
    detector = DistortionDetector()
    visualizer = Visualizer()
    
    # 2. Load Data
    print(f"Loading data (Limit: {args.rows} rows)...")
    df = loader.load_data(posts_path=args.posts_path, comments_path=args.comments_path, nrows=args.rows)
    
    if df.empty:
        print("No data found! Please check data/raw/ or provide paths.")
        return

    # 3. Preprocess
    print("Preprocessing sentences...")
    sentences_df = loader.preprocess_sentences(df)
    print(f"Total Sentences: {len(sentences_df)}")
    
    # 4. Detect Distortions
    print("Detecting cognitive distortions...")
    result_df, distortion_names = detector.detect(sentences_df)
    
    # Save intermediate result
    output_path = os.path.join(Config.PROCESSED_DATA_DIR, Config.DISTORTION_DATA_FILENAME)
    result_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    # 5. Visualization
    if args.mode == 'all':
        print("Generating Visualizations...")
        visualizer.plot_time_series(result_df, distortion_names)
        visualizer.plot_correlation_matrix(result_df, distortion_names)
        
    # 6. Topic Modeling (Optional or if specialized mode)
    # Only run if explicitly asked or if 'all' includes it (might be slow for 'all')
    if args.mode == 'topic_model':
        modeler = TopicModeler()
        for distortion in distortion_names:
            print(f"Running Topic Modeling for {distortion}...")
            clustered_df = modeler.run_clustering(result_df, distortion)
            if clustered_df is not None:
                cluster_path = os.path.join(Config.TABLES_DIR, f'topics_{distortion.replace(" ", "_")}.csv')
                clustered_df.to_csv(cluster_path, index=False)
                print(f"Saved clusters to {cluster_path}")

if __name__ == "__main__":
    main()
