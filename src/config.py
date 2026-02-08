import os

class Config:
    # Base Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
    PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
    PLOT_CORR_DIR = os.path.join(PLOTS_DIR, 'correlation')
    PLOT_TS_RAW_DIR = os.path.join(PLOTS_DIR, 'time_series', 'raw')
    PLOT_TS_NORM_DIR = os.path.join(PLOTS_DIR, 'time_series', 'normalized')
    PLOT_TS_SPIKES_DIR = os.path.join(PLOTS_DIR, 'time_series', 'spikes')
    TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
    
    # File Names (Expected Input)
    POSTS_FILENAME = 'posts.csv'
    COMMENTS_FILENAME = 'comments.csv'
    
    # Output Filenames
    MERGED_DATA_FILENAME = 'merged_data.csv'
    DISTORTION_DATA_FILENAME = 'distortion_data.csv'

    # Columns
    TEXT_COLUMN = 'text'
    DATE_COLUMN = 'date'
    AUTHOR_COLUMN = 'author'
    
    # Model
    MODEL_NAME = 'all-mpnet-base-v2'
    
    # Analysis
    CLUSTERS_K_MIN = 10
    CLUSTERS_K_MAX = 100
    CLUSTERS_K_STEP = 10
    
    # Dates
    COVID_START_DATE = '2020-04-07'
    COVID_END_DATE = '2022-01-01'
    
    @staticmethod
    def ensure_directories():
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.TABLES_DIR, exist_ok=True)
        os.makedirs(Config.PLOT_CORR_DIR, exist_ok=True)
        os.makedirs(Config.PLOT_TS_RAW_DIR, exist_ok=True)
        os.makedirs(Config.PLOT_TS_NORM_DIR, exist_ok=True)
        os.makedirs(Config.PLOT_TS_SPIKES_DIR, exist_ok=True)
