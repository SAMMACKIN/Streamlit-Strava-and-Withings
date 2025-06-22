# Health Data Analyzer

A comprehensive Streamlit application for analyzing your health and fitness data from multiple sources.

## Overview

This application integrates with Withings health devices and Strava fitness data to provide comprehensive health analytics, correlations, and insights using machine learning models.

## Features

### 🏥 Withings Health Data
- **Weight Tracking**: Monitor weight trends with moving averages
- **Sleep Analysis**: Deep sleep, REM sleep, and sleep quality scoring
- **Body Composition**: Fat %, muscle mass, BMI, and detailed body metrics
- **Health Dashboard**: Comprehensive overview of all health metrics

### 🏃 Strava Exercise Data  
- **Exercise Analytics**: Distance, pace, elevation trends and seasonal patterns
- **Performance Metrics**: Year-over-year comparisons and activity distributions
- **Training Analysis**: Multi-year exercise patterns and intensity heatmaps

### 🔗 Advanced Correlations
- **Exercise-Sleep Impact**: ML-powered analysis of how exercise affects sleep quality
- **Health Correlations**: Statistical analysis between weight, sleep, and exercise
- **Predictive Insights**: Regression models for sleep quality prediction

## Setup

1. **Activate virtual environment:**
   ```bash
   source strava_env/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API credentials:**
   - Create `.env` file with Withings and Strava API credentials
   - For Streamlit Cloud: Use Secrets management in the dashboard

4. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## API Configuration

### Local Development (.env file)
```bash
WITHINGS_CLIENT_ID=your_withings_client_id
WITHINGS_CLIENT_SECRET=your_withings_client_secret
WITHINGS_REDIRECT_URI=http://localhost:8501

STRAVA_CLIENT_ID=your_strava_client_id  
STRAVA_CLIENT_SECRET=your_strava_client_secret
STRAVA_REDIRECT_URI=http://localhost:8501
```

### Streamlit Cloud (Secrets)
Add the same variables in your Streamlit Cloud app's Secrets section with your production URLs.

## Technologies

- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualizations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models for correlation analysis
- **OAuth2**: Secure API authentication for Withings and Strava