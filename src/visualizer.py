import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from .config import Config

class Visualizer:
    def __init__(self):
        # Set simplistic style
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_time_series(self, df, distortion_names, period='W'):
        """
        Resamples data by `period` (default Weekly), normalizes it, and plots trends.
        """
        print("Generating Time Series Plots...")
        df = df.dropna(subset=[Config.DATE_COLUMN])
        df = df.set_index(Config.DATE_COLUMN).sort_index()

        # Total count per period (normalization factor)
        total_counts = df.resample(period).size()
        
        for distortion in distortion_names:
            # Count distortion
            d_counts = df[distortion].resample(period).sum()
            
            # Normalize (percentage)
            normalized = (d_counts / total_counts) * 100
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(normalized.index, normalized.values, label=f'{distortion} (%)')
            
            # Simple moving average (4 periods)
            ma = normalized.rolling(window=4).mean()
            plt.plot(normalized.index, ma, label='4-Week MA', linestyle='--', color='orange')
            
            plt.title(f'Trend of {distortion} over Time')
            plt.xlabel('Date')
            plt.ylabel('Percentage of Sentences')
            plt.legend()
            
            filename = f"trend_{distortion.replace(' ', '_')}.png"
            path = os.path.join(Config.PLOTS_DIR, filename)
            plt.savefig(path)
            plt.close()
            print(f"Saved {filename}")

    def plot_correlation_matrix(self, df, distortion_names):
        """
        Calculates correlation between distortions and saves heatmap.
        """
        print("Generating Correlation Matrix...")
        
        # We need per-comment or per-post correlation, or time-series correlation
        # The user's notebook did time-series correlation. Let's do that.
        
        df = df.dropna(subset=[Config.DATE_COLUMN])
        df = df.set_index(Config.DATE_COLUMN).sort_index()
        
        # Resample to weekly sums for correlation
        weekly_sums = df[distortion_names].resample('W').sum()
        # Normalize
        weekly_totals = df.resample('W').size()
        weekly_norm = weekly_sums.div(weekly_totals, axis=0)
        
        corr = weekly_norm.corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation of Cognitive Distortions (Weekly Trends)')
        plt.tight_layout()
        
        path = os.path.join(Config.PLOTS_DIR, 'correlation_matrix.png')
        plt.savefig(path)
        plt.close()
        print(f"Saved correlation_matrix.png")
        return corr
