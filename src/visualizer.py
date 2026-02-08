import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
from .config import Config

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')

    def filter_and_identify_spikes(self, data, window_size=4):
        """
        Detects spikes in a time series.
        """
        data_arr = np.array(data)
        filtered_data = []
        
        for i in range(len(data_arr)):
            if i >= window_size:
                window = data_arr[i - window_size:i]
                avg_prev = np.mean(window)
                std_dev = np.std(window)
                if data_arr[i] - avg_prev > std_dev:
                    filtered_data.append(data_arr[i])
                else:
                    filtered_data.append(0)
            else:
                filtered_data.append(0)
        return pd.Series(filtered_data, index=data.index)

    def prepare_time_series(self, df, distortion_names):
        """
        Aggregates data into weekly Raw, Normalized, and Spike series.
        """
        df = df.dropna(subset=[Config.DATE_COLUMN])
        df = df.set_index(Config.DATE_COLUMN).sort_index()
        
        weekly_data = {}
        if Config.AUTHOR_COLUMN in df.columns:
            weekly_posters = df[Config.AUTHOR_COLUMN].resample('W').nunique()
        else:
            weekly_posters = df.resample('W').size()
        weekly_posters = weekly_posters.replace(0, 1)

        for distortion in distortion_names:
            raw_counts = df[distortion].resample('W').sum()
            norm_counts = (raw_counts / weekly_posters) * 100
            spikes = self.filter_and_identify_spikes(norm_counts)
            
            weekly_data[distortion] = {
                'raw': raw_counts,
                'norm': norm_counts,
                'spikes': spikes
            }
        return weekly_data

    def _plot_grid(self, weekly_data, data_type, title_suffix, output_dir, color):
        """
        Helper to plot 12 distortions in a single figure (3x4 grid) AND individual plots.
        """
        distortion_names = list(weekly_data.keys())
        n = len(distortion_names)
        cols = 4
        rows = math.ceil(n / cols)
        
        # 1. Summary Plot (All in One Image)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, distortion in enumerate(distortion_names):
            series = weekly_data[distortion][data_type]
            ax = axes[i]
            if data_type == 'spikes':
                ax.bar(series.index, series, color=color, width=5)
            else:
                ax.plot(series.index, series, color=color)
            ax.set_title(distortion)
            ax.tick_params(axis='x', rotation=45)
            
            # 2. Individual Plot (One per file)
            plt.figure(figsize=(10, 6))
            if data_type == 'spikes':
                plt.bar(series.index, series, color=color, width=5)
            else:
                plt.plot(series.index, series, color=color)
            plt.title(f'{distortion} - {title_suffix}')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.tight_layout()
            indiv_path = os.path.join(output_dir, f'{distortion.replace(" ", "_")}.png')
            plt.savefig(indiv_path)
            plt.close()

        # Hide empty subplots in summary
        for i in range(n, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle(f'All Distortions - {title_suffix}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        summary_path = os.path.join(output_dir, 'SUMMARY_ALL_PLOTS.png')
        fig.savefig(summary_path)
        plt.close(fig)
        print(f"Saved summary and individual plots to {output_dir}")

    def plot_time_series(self, weekly_data):
        print("Generating Segregated Time Series Plots...")
        self._plot_grid(weekly_data, 'raw', 'Raw Counts', Config.PLOT_TS_RAW_DIR, 'blue')
        self._plot_grid(weekly_data, 'norm', 'Normalized', Config.PLOT_TS_NORM_DIR, 'green')
        self._plot_grid(weekly_data, 'spikes', 'Spikes', Config.PLOT_TS_SPIKES_DIR, 'red')

    def plot_combined_trends(self, weekly_data):
        print("Generating Combined Suman Plot...")
        plt.figure(figsize=(16, 8))
        for distortion, series_dict in weekly_data.items():
            smoothed = series_dict['norm'].rolling(window=4).mean()
            plt.plot(smoothed.index, smoothed, label=distortion, alpha=0.7)
            
        covid_start = pd.to_datetime(Config.COVID_START_DATE)
        covid_end = pd.to_datetime(Config.COVID_END_DATE)
        plt.axvline(x=covid_start, color='red', linestyle='--', label='COVID Start')
        plt.axvline(x=covid_end, color='red', linestyle='--', label='COVID End')
        
        plt.title('Combined Normalized Trends')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        path = os.path.join(Config.PLOTS_DIR, 'combined_trends_covid.png')
        plt.savefig(path)
        plt.close()

    def plot_correlation_matrices(self, weekly_data):
        print("Generating Time-Series Correlation Matrices...")
        periods = {
            'Before': (None, Config.COVID_START_DATE),
            'During': (Config.COVID_START_DATE, Config.COVID_END_DATE),
            'After': (Config.COVID_END_DATE, None)
        }
        data_types = ['raw', 'norm', 'spikes']
        
        for p_name, (start, end) in periods.items():
            for d_type in data_types:
                df = pd.DataFrame()
                for dist, s_dict in weekly_data.items():
                    series = s_dict[d_type]
                    if start: series = series[series.index >= pd.to_datetime(start)]
                    if end: series = series[series.index < pd.to_datetime(end)]
                    df[dist] = series
                
                if df.empty: continue
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                plt.title(f'Correlation: {p_name} ({d_type})')
                plt.tight_layout()
                path = os.path.join(Config.PLOT_CORR_DIR, f'corr_ts_{p_name}_{d_type}.png')
                plt.savefig(path)
                plt.close()

    def plot_per_comment_correlations(self, df, distortion_names):
        """
        Calculates correlation based on co-occurrence in COMMENTS only.
        Splits by time period.
        """
        print("Generating Per-Comment Correlation Matrices...")
        
        # Filter for comments only
        comments_df = df[df['source_type'] == 'comment'].copy()
        
        if comments_df.empty:
            print("No comments found for per-comment correlation.")
            return

        periods = {
            'Before': (None, Config.COVID_START_DATE),
            'During': (Config.COVID_START_DATE, Config.COVID_END_DATE),
            'After': (Config.COVID_END_DATE, None)
        }
        
        for p_name, (start, end) in periods.items():
            subset = comments_df.copy()
            if start:
                subset = subset[subset[Config.DATE_COLUMN] >= pd.to_datetime(start)]
            if end:
                subset = subset[subset[Config.DATE_COLUMN] < pd.to_datetime(end)]
            
            if subset.empty:
                continue
                
            # Convert boolean columns to int for correlation
            corr_df = subset[distortion_names].astype(int)
            corr = corr_df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title(f'Correlation (Per Comment): {p_name} COVID')
            plt.tight_layout()
            
            path = os.path.join(Config.PLOT_CORR_DIR, f'corr_comment_{p_name}.png')
            plt.savefig(path)
            plt.close()
            print(f"Saved corr_comment_{p_name}.png (n={len(subset)})")
