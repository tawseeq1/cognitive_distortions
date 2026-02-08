# Cognitive Distortion Analysis Pipeline

## Overview
This project analyzes cognitive distortions in Reddit posts and comments using N-gram detection, SentenceBERT embeddings, and Clustering.

## Setup
1. **Dependencies**: Ensure you have the required Python packages installed.
   ```bash
   pip install pandas numpy nltk scikit-learn sentence-transformers matplotlib seaborn
   ```

## Directory Structure
```
cog_dis/
├── main.py                     # CLI Entry Point
├── src/                        # Source Code
│   ├── config.py               # Configuration
│   ├── data_loader.py          # Data Ingestion
│   ├── distortion_detector.py  # N-gram Logic
│   ├── topic_modeler.py        # Clustering
│   └── visualizer.py           # Plotting
└── data/                       # Data Storage
    ├── raw/                    # Place posts.csv and comments.csv here
    ├── processed/              # Intermediate files
    └── output/                 # Results
        ├── plots/
        └── tables/
```

## Usage

### 1. Prepare Data
Place your `posts.csv` and `comments.csv` files in the `data/raw/` directory.

### 2. Run the Pipeline
Run the main script from the root directory:

```bash
# Run on all data
python main.py

# Run on a small sample (e.g., 100 rows) for testing
python main.py --rows 100

# Run specific analysis mode (e.g., topic modeling)
python main.py --mode topic_model
```

### 3. Output
- **Plots**: Time series and Correlation Heatmaps will be saved in `data/output/plots/`.
- **Processed Data**: `distortion_data.csv` in `data/processed/`.
