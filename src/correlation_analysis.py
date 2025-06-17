import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from datetime import datetime, timedelta

class CorrelationAnalysis:
    def __init__(self):
        pass
    
    def prepare_combined_data(self, strava_df, withings_weight_df, withings_sleep_df):
        """Combine and align data from different sources by date"""
        # Prepare Strava data (daily aggregation)
        if not strava_df.empty:
            strava_daily = strava_df.groupby(strava_df['start_date_local'].dt.date).agg({
                'distance_km': 'sum',
                'moving_time_hours': 'sum',
                'total_elevation_gain': 'sum',
                'id': 'count'
            }).reset_index()
            strava_daily.columns = ['date', 'daily_distance', 'daily_exercise_time', 'daily_elevation', 'daily_activities']
            strava_daily['date'] = pd.to_datetime(strava_daily['date'])
        else:
            strava_daily = pd.DataFrame()
        
        # Prepare weight data (daily - take most recent measurement per day)
        if not withings_weight_df.empty:
            weight_daily = withings_weight_df.copy()
            weight_daily['date_key'] = weight_daily['date'].dt.date
            weight_daily = weight_daily.groupby('date_key').last().reset_index()
            weight_daily['date'] = pd.to_datetime(weight_daily['date_key'])
            weight_daily = weight_daily.drop('date_key', axis=1)
        else:
            weight_daily = pd.DataFrame()
        
        # Prepare sleep data (daily)
        if not withings_sleep_df.empty:
            sleep_daily = withings_sleep_df.copy()
            sleep_daily['date_key'] = sleep_daily['date'].dt.date
            
            # Aggregate sleep data by date
            sleep_agg = sleep_daily.groupby('date_key').agg({
                'sleep_duration': 'sum',
                'deep_sleep': 'sum',
                'light_sleep': 'sum', 
                'rem_sleep': 'sum',
                'sleep_score': 'mean'
            }).reset_index()
            
            sleep_agg['date'] = pd.to_datetime(sleep_agg['date_key'])
            sleep_daily = sleep_agg.drop('date_key', axis=1)
        else:
            sleep_daily = pd.DataFrame()
        
        # Merge all data systematically
        all_dataframes = []
        
        if not strava_daily.empty:
            all_dataframes.append(strava_daily)
        if not weight_daily.empty:
            all_dataframes.append(weight_daily)
        if not sleep_daily.empty:
            all_dataframes.append(sleep_daily)
        
        if not all_dataframes:
            return pd.DataFrame()
        
        # Start with first dataframe and merge others
        combined_df = all_dataframes[0].copy()
        
        for df in all_dataframes[1:]:
            combined_df = pd.merge(combined_df, df, on='date', how='outer', suffixes=('', '_duplicate'))
            
            # Remove duplicate columns that might have been created
            duplicate_cols = [col for col in combined_df.columns if col.endswith('_duplicate')]
            combined_df = combined_df.drop(columns=duplicate_cols)
        
        # Sort by date and fill forward for weight (since weight changes gradually)
        if not combined_df.empty:
            combined_df = combined_df.sort_values('date')
            if 'weight' in combined_df.columns:
                combined_df['weight'] = combined_df['weight'].fillna(method='ffill')
        
        return combined_df
    
    def calculate_correlations(self, df):
        """Calculate correlation coefficients between different metrics"""
        if df.empty:
            return {}
        
        correlations = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Define interesting correlation pairs
        correlation_pairs = [
            ('daily_exercise_time', 'weight'),
            ('daily_exercise_time', 'sleep_duration'),
            ('daily_exercise_time', 'deep_sleep'),
            ('daily_exercise_time', 'sleep_score'),
            ('daily_distance', 'weight'),
            ('daily_distance', 'sleep_duration'),
            ('sleep_duration', 'weight'),
            ('deep_sleep', 'weight'),
            ('sleep_score', 'weight'),
            ('daily_activities', 'sleep_duration'),
            ('daily_elevation', 'sleep_duration')
        ]
        
        for col1, col2 in correlation_pairs:
            if col1 in numeric_columns and col2 in numeric_columns:
                # Remove rows where either value is NaN
                valid_data = df[[col1, col2]].dropna()
                if len(valid_data) > 10:  # Need at least 10 data points
                    correlation, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                    correlations[f"{col1}_vs_{col2}"] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'sample_size': len(valid_data),
                        'significant': p_value < 0.05
                    }
        
        return correlations
    
    def create_correlation_matrix(self, df):
        """Create correlation matrix visualization"""
        if df.empty:
            return None
        
        # Select numeric columns for correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Filter out columns with too many NaN values
        valid_cols = []
        for col in numeric_cols:
            if df[col].count() / len(df) > 0.3:  # At least 30% non-null values
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            return None
        
        correlation_matrix = df[valid_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Correlation Matrix: Exercise, Sleep & Weight',
            labels={'color': 'Correlation Coefficient'},
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        return fig
    
    def create_time_series_comparison(self, df):
        """Create time series plots comparing different metrics"""
        if df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Exercise Time vs Weight', 'Sleep Duration vs Weight', 'Exercise vs Sleep'],
            shared_xaxes=True
        )
        
        # Plot 1: Exercise Time vs Weight
        if 'daily_exercise_time' in df.columns and 'weight' in df.columns:
            # Exercise time
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['daily_exercise_time'],
                    name='Exercise Time (hrs)',
                    line=dict(color='blue'),
                    yaxis='y1'
                ),
                row=1, col=1
            )
            
            # Weight (on secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['weight'],
                    name='Weight (kg)',
                    line=dict(color='red'),
                    yaxis='y2'
                ),
                row=1, col=1
            )
        
        # Plot 2: Sleep Duration vs Weight
        if 'sleep_duration' in df.columns and 'weight' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sleep_duration'],
                    name='Sleep Duration (hrs)',
                    line=dict(color='green'),
                    yaxis='y3'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['weight'],
                    name='Weight (kg)',
                    line=dict(color='red'),
                    yaxis='y4',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 3: Exercise vs Sleep
        if 'daily_exercise_time' in df.columns and 'sleep_duration' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['daily_exercise_time'],
                    name='Exercise Time (hrs)',
                    line=dict(color='blue'),
                    yaxis='y5',
                    showlegend=False
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sleep_duration'],
                    name='Sleep Duration (hrs)',
                    line=dict(color='green'),
                    yaxis='y6',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            title_text="Time Series Comparison of Exercise, Sleep & Weight"
        )
        
        return fig
    
    def create_scatter_plots(self, df):
        """Create scatter plots for key correlations"""
        if df.empty:
            return []
        
        plots = []
        
        # Exercise vs Weight
        if 'daily_exercise_time' in df.columns and 'weight' in df.columns:
            clean_data = df[['daily_exercise_time', 'weight']].dropna()
            if len(clean_data) > 5:
                fig = px.scatter(
                    clean_data,
                    x='daily_exercise_time',
                    y='weight',
                    title='Exercise Time vs Weight',
                    labels={'daily_exercise_time': 'Daily Exercise Time (hours)', 'weight': 'Weight (kg)'},
                    trendline='ols'
                )
                plots.append(fig)
        
        # Sleep vs Weight
        if 'sleep_duration' in df.columns and 'weight' in df.columns:
            clean_data = df[['sleep_duration', 'weight']].dropna()
            if len(clean_data) > 5:
                fig = px.scatter(
                    clean_data,
                    x='sleep_duration',
                    y='weight',
                    title='Sleep Duration vs Weight',
                    labels={'sleep_duration': 'Sleep Duration (hours)', 'weight': 'Weight (kg)'},
                    trendline='ols'
                )
                plots.append(fig)
        
        # Exercise vs Sleep
        if 'daily_exercise_time' in df.columns and 'sleep_duration' in df.columns:
            clean_data = df[['daily_exercise_time', 'sleep_duration']].dropna()
            if len(clean_data) > 5:
                fig = px.scatter(
                    clean_data,
                    x='daily_exercise_time',
                    y='sleep_duration',
                    title='Exercise Time vs Sleep Duration',
                    labels={'daily_exercise_time': 'Daily Exercise Time (hours)', 'sleep_duration': 'Sleep Duration (hours)'},
                    trendline='ols'
                )
                plots.append(fig)
        
        return plots
    
    def generate_insights(self, correlations, df):
        """Generate text insights from correlation analysis"""
        insights = []
        
        if not correlations:
            insights.append("Not enough data available for correlation analysis.")
            return insights
        
        # Analyze significant correlations
        significant_correlations = {k: v for k, v in correlations.items() if v['significant']}
        
        if significant_correlations:
            insights.append("**Significant Correlations Found:**")
            
            for name, stats in significant_correlations.items():
                correlation = stats['correlation']
                col1, col2 = name.replace('_vs_', ' and ').replace('_', ' ').title().split(' And ')
                
                if abs(correlation) > 0.5:
                    strength = "strong"
                elif abs(correlation) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                direction = "positive" if correlation > 0 else "negative"
                
                insights.append(f"- {col1} and {col2}: {strength} {direction} correlation ({correlation:.3f})")
        
        # Data availability insights
        if not df.empty:
            insights.append("\n**Data Availability:**")
            if 'daily_exercise_time' in df.columns:
                exercise_days = df['daily_exercise_time'].notna().sum()
                insights.append(f"- Exercise data: {exercise_days} days")
            
            if 'weight' in df.columns:
                weight_measurements = df['weight'].notna().sum()
                insights.append(f"- Weight measurements: {weight_measurements} days")
            
            if 'sleep_duration' in df.columns:
                sleep_days = df['sleep_duration'].notna().sum()
                insights.append(f"- Sleep data: {sleep_days} days")
        
        return insights