import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from .config import Config

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')

    def filter_and_identify_spikes(self, data, window_size=4):
        """
        Detects spikes in a time series based on moving average and standard deviation.
        Returns a series with spikes preserved and non-spikes set to 0.
        """
        # Convert to numpy for performance
        data_arr = np.array(data)
        filtered_data = []
        
        for i in range(len(data_arr)):
            if i >= window_size:
                window = data_arr[i - window_size:i]
                avg_prev = np.mean(window)
                std_dev = np.std(window)
                
                # Condition: Value > Moving Average + 1 SD
                if data_arr[i] - avg_prev > std_dev:
                    filtered_data.append(data_arr[i])
                else:
                    filtered_data.append(0)
            else:
                filtered_data.append(0) # Initial padding
                
        return pd.Series(filtered_data, index=data.index)

    def prepare_time_series(self, df, distortion_names):
        """
        Aggregates data into weekly Raw, Normalized, and Spike series.
        """
        df = df.dropna(subset=[Config.DATE_COLUMN])
        df = df.set_index(Config.DATE_COLUMN).sort_index()
        
        weekly_data = {}
        
        # 1. Total Posters (Unique Authors) per week for normalization
        # Note: If 'author' is missing, fallback to count of posts
        if Config.AUTHOR_COLUMN in df.columns:
            weekly_posters = df[Config.AUTHOR_COLUMN].resample('W').nunique()
        else:
            weekly_posters = df.resample('W').size()
            
        # Avoid division by zero
        weekly_posters = weekly_posters.replace(0, 1)

        for distortion in distortion_names:
            # A. Raw Counts
            raw_counts = df[distortion].resample('W').sum()
            
            # B. Normalized (Distortions per 100 Posters)
            norm_counts = (raw_counts / weekly_posters) * 100
            
            # C. Spikes (on Normalized data)
            spikes = self.filter_and_identify_spikes(norm_counts)
            
            weekly_data[distortion] = {
                'raw': raw_counts,
                'norm': norm_counts,
                'spikes': spikes
            }
            
        return weekly_data

    def plot_time_series(self, weekly_data):
        """
        Plots trends for Raw, Normalized, and Spikes for each distortion.
        """
        print("Generating individual time series plots...")
        
        for distortion, series_dict in weekly_data.items():
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # 1. Raw
            axes[0].plot(series_dict['raw'].index, series_dict['raw'], color='blue')
            axes[0].set_title(f'{distortion} - Raw Counts')
            
            # 2. Normalized
            axes[1].plot(series_dict['norm'].index, series_dict['norm'], color='green')
            axes[1].set_title(f'{distortion} - Normalized (per 100 posters)')
            
            # 3. Spikes
            axes[2].bar(series_dict['spikes'].index, series_dict['spikes'], color='red', width=5) # width in days approx
            axes[2].set_title(f'{distortion} - Spikes Only')
            
            plt.tight_layout()
            path = os.path.join(Config.PLOTS_DIR, f'series_{distortion.replace(" ", "_")}.png')
            plt.savefig(path)
            plt.close()

    def plot_combined_trends(self, weekly_data):
        """
        "Suman Plot": All distortions on one plot (Normalized), with COVID markers.
        """
        print("Generating Combined (Suman) Plot...")
        plt.figure(figsize=(16, 8))
        
        for distortion, series_dict in weekly_data.items():
            # Smooth lines for readability
            smoothed = series_dict['norm'].rolling(window=4).mean()
            plt.plot(smoothed.index, smoothed, label=distortion, alpha=0.7)
            
        # COVID Markers
        covid_start = pd.to_datetime(Config.COVID_START_DATE)
        covid_end = pd.to_datetime(Config.COVID_END_DATE)
        
        plt.axvline(x=covid_start, color='red', linestyle='--', linewidth=2, label='COVID Start')
        plt.axvline(x=covid_end, color='red', linestyle='--', linewidth=2, label='COVID End')
        
        plt.title('Combined Normalized Trends of Cognitive Distortions (4-Week Rolling Avg)')
        plt.xlabel('Date')
        plt.ylabel('Frequency (per 100 posters)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        path = os.path.join(Config.PLOTS_DIR, 'combined_trends_covid.png')
        plt.savefig(path)
        plt.close()

    def plot_correlation_matrices(self, weekly_data):
        """
        Generates 9 Correlation Matrices:
        Periods: Before, During, After COVID
        Types: Raw, Norm, Spikes
        """
        print("Generating Correlation Matrices...")
        
        periods = {
            'Before': (None, Config.COVID_START_DATE),
            'During': (Config.COVID_START_DATE, Config.COVID_END_DATE),
            'After': (Config.COVID_END_DATE, None)
        }
        
        data_types = ['raw', 'norm', 'spikes']
        
        for p_name, (start, end) in periods.items():
            for d_type in data_types:
                # 1. Build DataFrame for this period & type
                combined_df = pd.DataFrame()
                
                for distortion, series_dict in weekly_data.items():
                    series = series_dict[d_type]
                    
                    # Filter by date
                    if start:
                        series = series[series.index >= pd.to_datetime(start)]
                    if end:
                        series = series[series.index < pd.to_datetime(end)]
                        
                    combined_df[distortion] = series
                
                if combined_df.empty:
                    print(f"Warning: No data for {p_name} ({d_type}). Skipping.")
                    continue
                    
                # 2. Calculate Correlation
                corr = combined_df.corr()
                
                # 3. Plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
                plt.title(f'Correlation: {p_name} COVID ({d_type.capitalize()})')
                plt.tight_layout()
                
                filename = f'corr_{p_name}_{d_type}.png'
                path = os.path.join(Config.PLOTS_DIR, filename)
                plt.savefig(path)
                plt.close()
                print(f"Saved {filename}")
