import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import sklearn components with fallback
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"Sklearn not available: {e}")

# Alternative simple implementations if sklearn fails
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: (X'X)^-1 X'y
        try:
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
        except:
            self.intercept_ = 0
            self.coef_ = np.zeros(X.shape[1])
    
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_

def simple_r2_score(y_true, y_pred):
    """Simple R² calculation"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0
    
    return 1 - (ss_res / ss_tot)

class ExerciseSleepAnalysis:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
    
    def prepare_exercise_sleep_data(self, strava_df, sleep_df):
        """Prepare data for exercise-sleep impact analysis"""
        if strava_df.empty or sleep_df.empty:
            return pd.DataFrame()
        
        # Prepare exercise data with timing
        exercise_data = strava_df.copy()
        exercise_data['exercise_date'] = exercise_data['start_date_local'].dt.date
        exercise_data['exercise_hour'] = exercise_data['start_date_local'].dt.hour
        exercise_data['exercise_time_category'] = exercise_data['exercise_hour'].apply(self._categorize_exercise_time)
        
        # Daily exercise aggregation
        daily_exercise = exercise_data.groupby('exercise_date').agg({
            'moving_time_hours': 'sum',
            'distance_km': 'sum',
            'total_elevation_gain': 'sum',
            'exercise_hour': ['mean', 'min', 'max'],  # Average and range of exercise times
            'exercise_time_category': lambda x: x.mode().iloc[0] if not x.empty else 'None'
        }).reset_index()
        
        # Flatten column names
        daily_exercise.columns = ['exercise_date', 'total_exercise_time', 'total_distance', 
                                'total_elevation', 'avg_exercise_hour', 'earliest_exercise', 
                                'latest_exercise', 'primary_exercise_time']
        
        # Prepare sleep data
        sleep_data = sleep_df.copy()
        sleep_data['sleep_date'] = sleep_data['date'].dt.date
        
        # Create features for next-day and two-day sleep analysis
        analysis_data = []
        
        for idx, exercise_row in daily_exercise.iterrows():
            exercise_date = exercise_row['exercise_date']
            
            # Find sleep data for next 1-2 nights
            night1_date = exercise_date + timedelta(days=1)
            night2_date = exercise_date + timedelta(days=2)
            
            night1_sleep = sleep_data[sleep_data['sleep_date'] == night1_date]
            night2_sleep = sleep_data[sleep_data['sleep_date'] == night2_date]
            
            # Create record with exercise and subsequent sleep data
            record = {
                'exercise_date': exercise_date,
                'total_exercise_time': exercise_row['total_exercise_time'],
                'total_distance': exercise_row['total_distance'],
                'total_elevation': exercise_row['total_elevation'],
                'avg_exercise_hour': exercise_row['avg_exercise_hour'],
                'earliest_exercise': exercise_row['earliest_exercise'],
                'latest_exercise': exercise_row['latest_exercise'],
                'primary_exercise_time': exercise_row['primary_exercise_time'],
                'exercise_intensity': self._calculate_intensity(exercise_row)
            }
            
            # Night 1 sleep (night after exercise)
            if not night1_sleep.empty:
                night1 = night1_sleep.iloc[0]
                record.update({
                    'night1_sleep_duration': night1.get('sleep_duration', np.nan),
                    'night1_deep_sleep': night1.get('deep_sleep', np.nan),
                    'night1_rem_sleep': night1.get('rem_sleep', np.nan),
                    'night1_sleep_score': night1.get('sleep_score', np.nan),
                    'night1_sleep_efficiency': self._calculate_sleep_efficiency(night1)
                })
            else:
                record.update({
                    'night1_sleep_duration': np.nan,
                    'night1_deep_sleep': np.nan,
                    'night1_rem_sleep': np.nan,
                    'night1_sleep_score': np.nan,
                    'night1_sleep_efficiency': np.nan
                })
            
            # Night 2 sleep (second night after exercise)
            if not night2_sleep.empty:
                night2 = night2_sleep.iloc[0]
                record.update({
                    'night2_sleep_duration': night2.get('sleep_duration', np.nan),
                    'night2_deep_sleep': night2.get('deep_sleep', np.nan),
                    'night2_rem_sleep': night2.get('rem_sleep', np.nan),
                    'night2_sleep_score': night2.get('sleep_score', np.nan),
                    'night2_sleep_efficiency': self._calculate_sleep_efficiency(night2)
                })
            else:
                record.update({
                    'night2_sleep_duration': np.nan,
                    'night2_deep_sleep': np.nan,
                    'night2_rem_sleep': np.nan,
                    'night2_sleep_score': np.nan,
                    'night2_sleep_efficiency': np.nan
                })
            
            analysis_data.append(record)
        
        df = pd.DataFrame(analysis_data)
        
        # Add baseline sleep metrics (average sleep when no exercise previous day)
        df = self._add_baseline_sleep_metrics(df, sleep_data)
        
        return df
    
    def _categorize_exercise_time(self, hour):
        """Categorize exercise time into periods"""
        if 5 <= hour < 10:
            return 'Early Morning'
        elif 10 <= hour < 14:
            return 'Late Morning'
        elif 14 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    
    def _calculate_intensity(self, exercise_row):
        """Calculate exercise intensity score"""
        # Normalize metrics and combine
        time_factor = min(exercise_row['total_exercise_time'] / 2.0, 1.0)  # Cap at 2 hours
        distance_factor = min(exercise_row['total_distance'] / 20.0, 1.0)  # Cap at 20km
        elevation_factor = min(exercise_row['total_elevation'] / 1000.0, 1.0)  # Cap at 1000m
        
        return (time_factor + distance_factor + elevation_factor) / 3.0
    
    def _calculate_sleep_efficiency(self, sleep_row):
        """Calculate sleep efficiency metric"""
        if pd.isna(sleep_row.get('sleep_duration')) or sleep_row.get('sleep_duration', 0) == 0:
            return np.nan
        
        deep_sleep = sleep_row.get('deep_sleep', 0)
        rem_sleep = sleep_row.get('rem_sleep', 0)
        total_sleep = sleep_row.get('sleep_duration', 1)
        
        # Efficiency = (deep + REM) / total sleep time
        return (deep_sleep + rem_sleep) / total_sleep if total_sleep > 0 else np.nan
    
    def _add_baseline_sleep_metrics(self, df, sleep_data):
        """Add baseline sleep metrics for comparison"""
        # Calculate average sleep metrics for baseline
        baseline_metrics = {
            'baseline_sleep_duration': sleep_data['sleep_duration'].mean(),
            'baseline_deep_sleep': sleep_data['deep_sleep'].mean(),
            'baseline_rem_sleep': sleep_data['rem_sleep'].mean(),
            'baseline_sleep_score': sleep_data['sleep_score'].mean()
        }
        
        for metric, value in baseline_metrics.items():
            df[metric] = value
        
        return df
    
    def perform_regression_analysis(self, df):
        """Perform multiple regression analyses"""
        results = {}
        
        # Define target variables (sleep outcomes)
        sleep_targets = ['night1_sleep_duration', 'night1_deep_sleep', 'night1_sleep_score',
                        'night2_sleep_duration', 'night2_deep_sleep', 'night2_sleep_score']
        
        # Define predictor variables
        predictors = ['total_exercise_time', 'total_distance', 'exercise_intensity', 'avg_exercise_hour']
        
        for target in sleep_targets:
            if target in df.columns:
                # Prepare data (remove NaN values)
                analysis_df = df[predictors + [target]].dropna()
                
                if len(analysis_df) > 10:  # Need sufficient data
                    X = analysis_df[predictors]
                    y = analysis_df[target]
                    
                    try:
                        if SKLEARN_AVAILABLE:
                            # Linear regression with sklearn
                            lr_model = LinearRegression()
                            lr_model.fit(X, y)
                            lr_predictions = lr_model.predict(X)
                            lr_r2 = r2_score(y, lr_predictions)
                            
                            # Random Forest for non-linear relationships
                            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf_model.fit(X, y)
                            rf_predictions = rf_model.predict(X)
                            rf_r2 = r2_score(y, rf_predictions)
                            
                            results[target] = {
                                'linear_r2': lr_r2,
                                'rf_r2': rf_r2,
                                'linear_coefficients': dict(zip(predictors, lr_model.coef_)),
                                'feature_importance': dict(zip(predictors, rf_model.feature_importances_)),
                                'sample_size': len(analysis_df)
                            }
                        else:
                            # Fallback to simple linear regression
                            lr_model = SimpleLinearRegression()
                            lr_model.fit(X, y)
                            lr_predictions = lr_model.predict(X)
                            lr_r2 = simple_r2_score(y, lr_predictions)
                            
                            # Simple feature importance (correlation-based)
                            feature_importance = {}
                            for predictor in predictors:
                                corr = np.corrcoef(X[predictor], y)[0, 1]
                                feature_importance[predictor] = abs(corr) if not np.isnan(corr) else 0
                            
                            results[target] = {
                                'linear_r2': lr_r2,
                                'rf_r2': lr_r2,  # Use same as linear for fallback
                                'linear_coefficients': dict(zip(predictors, lr_model.coef_)),
                                'feature_importance': feature_importance,
                                'sample_size': len(analysis_df)
                            }
                    except Exception as e:
                        print(f"Error in regression for {target}: {e}")
                        continue
        
        return results
    
    def analyze_exercise_timing_effects(self, df):
        """Analyze how exercise timing affects sleep"""
        timing_analysis = {}
        
        # Group by exercise time category
        time_categories = ['Early Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night']
        sleep_metrics = ['night1_sleep_duration', 'night1_deep_sleep', 'night1_sleep_score']
        
        for metric in sleep_metrics:
            if metric in df.columns:
                category_effects = {}
                
                for category in time_categories:
                    category_data = df[df['primary_exercise_time'] == category][metric].dropna()
                    baseline = df[f'baseline_{metric.replace("night1_", "")}'].iloc[0] if len(df) > 0 else np.nan
                    
                    if len(category_data) > 2:
                        mean_sleep = category_data.mean()
                        effect_size = mean_sleep - baseline if not pd.isna(baseline) else np.nan
                        
                        category_effects[category] = {
                            'mean': mean_sleep,
                            'count': len(category_data),
                            'effect_size': effect_size,
                            'std': category_data.std()
                        }
                
                timing_analysis[metric] = category_effects
        
        return timing_analysis
    
    def create_exercise_timing_plot(self, df):
        """Create visualization of exercise timing effects"""
        if df.empty:
            return None
        
        # Calculate average sleep metrics by exercise time
        timing_summary = df.groupby('primary_exercise_time').agg({
            'night1_sleep_duration': 'mean',
            'night1_deep_sleep': 'mean',
            'night1_sleep_score': 'mean',
            'total_exercise_time': 'count'
        }).reset_index()
        
        timing_summary = timing_summary[timing_summary['total_exercise_time'] >= 3]  # At least 3 observations
        
        if timing_summary.empty:
            return None
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sleep Duration by Exercise Timing', 'Deep Sleep by Exercise Timing',
                          'Sleep Score by Exercise Timing', 'Sample Sizes'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sleep duration
        fig.add_trace(
            go.Bar(x=timing_summary['primary_exercise_time'], 
                   y=timing_summary['night1_sleep_duration'],
                   name='Sleep Duration (hrs)',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Deep sleep
        fig.add_trace(
            go.Bar(x=timing_summary['primary_exercise_time'], 
                   y=timing_summary['night1_deep_sleep'],
                   name='Deep Sleep (hrs)',
                   marker_color='darkblue'),
            row=1, col=2
        )
        
        # Sleep score
        fig.add_trace(
            go.Bar(x=timing_summary['primary_exercise_time'], 
                   y=timing_summary['night1_sleep_score'],
                   name='Sleep Score',
                   marker_color='green'),
            row=2, col=1
        )
        
        # Sample sizes
        fig.add_trace(
            go.Bar(x=timing_summary['primary_exercise_time'], 
                   y=timing_summary['total_exercise_time'],
                   name='# Observations',
                   marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Exercise Timing Effects on Sleep Quality',
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_exercise_sleep_scatter(self, df):
        """Create scatter plots showing exercise-sleep relationships"""
        if df.empty:
            return []
        
        plots = []
        
        # Exercise intensity vs sleep quality
        if 'exercise_intensity' in df.columns and 'night1_sleep_score' in df.columns:
            clean_data = df[['exercise_intensity', 'night1_sleep_score', 'primary_exercise_time']].dropna()
            
            if len(clean_data) > 5:
                fig = px.scatter(
                    clean_data,
                    x='exercise_intensity',
                    y='night1_sleep_score',
                    color='primary_exercise_time',
                    title='Exercise Intensity vs Sleep Score (Next Night)',
                    labels={'exercise_intensity': 'Exercise Intensity (0-1)', 
                           'night1_sleep_score': 'Sleep Score'},
                    trendline='ols'
                )
                plots.append(fig)
        
        # Exercise timing (hour) vs sleep duration
        if 'avg_exercise_hour' in df.columns and 'night1_sleep_duration' in df.columns:
            clean_data = df[['avg_exercise_hour', 'night1_sleep_duration']].dropna()
            
            if len(clean_data) > 5:
                fig = px.scatter(
                    clean_data,
                    x='avg_exercise_hour',
                    y='night1_sleep_duration',
                    title='Exercise Time of Day vs Sleep Duration (Next Night)',
                    labels={'avg_exercise_hour': 'Average Exercise Hour (24h)', 
                           'night1_sleep_duration': 'Sleep Duration (hours)'},
                    trendline='ols'
                )
                plots.append(fig)
        
        return plots
    
    def generate_insights(self, df, regression_results, timing_analysis):
        """Generate key insights from the analysis"""
        insights = []
        
        if df.empty:
            insights.append("❌ Insufficient data for exercise-sleep analysis.")
            return insights
        
        insights.append(f"📊 **Analysis Summary**: {len(df)} exercise sessions analyzed with sleep data.")
        
        # Timing insights
        if timing_analysis:
            best_times = []
            for metric, categories in timing_analysis.items():
                if categories:
                    best_category = max(categories.keys(), 
                                      key=lambda x: categories[x].get('effect_size', -999)
                                      if not pd.isna(categories[x].get('effect_size')) else -999)
                    best_effect = categories[best_category].get('effect_size', 0)
                    
                    if not pd.isna(best_effect):
                        metric_name = metric.replace('night1_', '').replace('_', ' ').title()
                        insights.append(f"🕐 **Best time for {metric_name}**: {best_category} "
                                      f"(+{best_effect:.2f} improvement)")
        
        # Regression insights
        if regression_results:
            for target, results in regression_results.items():
                if results['linear_r2'] > 0.1:  # Meaningful relationship
                    metric_name = target.replace('night1_', '').replace('night2_', '').replace('_', ' ').title()
                    night = "first" if "night1" in target else "second"
                    
                    # Find strongest predictor
                    strongest_predictor = max(results['feature_importance'].keys(),
                                            key=lambda x: results['feature_importance'][x])
                    
                    insights.append(f"📈 **{metric_name} ({night} night)**: "
                                  f"R² = {results['linear_r2']:.3f}, "
                                  f"strongest factor: {strongest_predictor.replace('_', ' ')}")
        
        # Exercise timing effect insights
        if 'avg_exercise_hour' in df.columns:
            early_exercise = df[df['avg_exercise_hour'] < 12]
            late_exercise = df[df['avg_exercise_hour'] >= 18]
            
            if len(early_exercise) > 2 and len(late_exercise) > 2:
                early_sleep = early_exercise['night1_sleep_score'].mean()
                late_sleep = late_exercise['night1_sleep_score'].mean()
                
                if not pd.isna(early_sleep) and not pd.isna(late_sleep):
                    difference = early_sleep - late_sleep
                    if difference > 5:
                        insights.append(f"🌅 **Early exercise benefit**: "
                                      f"Morning exercise leads to {difference:.1f} point higher sleep score")
                    elif difference < -5:
                        insights.append(f"🌙 **Evening exercise benefit**: "
                                      f"Evening exercise leads to {abs(difference):.1f} point higher sleep score")
        
        return insights