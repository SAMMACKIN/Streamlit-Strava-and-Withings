import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import os

from strava_auth import StravaAuth
from strava_api import StravaAPI
from withings_auth import WithingsAuth
from withings_api import WithingsAPI
from correlation_analysis import CorrelationAnalysis
from exercise_sleep_analysis import ExerciseSleepAnalysis
from token_storage import TokenStorage

# Page config
st.set_page_config(
    page_title="Strava Data Analyzer",
    page_icon="🏃",
    layout="wide"
)

# Initialize authentication and storage
auth = StravaAuth()
withings_auth = WithingsAuth()
token_storage = TokenStorage()

def main():
    st.title("📊 Health Data Analyzer")
    
    # Sidebar for navigation and authentication
    with st.sidebar:
        st.header("Navigation")
        
        # Check for authorization code first (before checking authentication)
        query_params = st.query_params
        
        # Load saved tokens on app start
        if not hasattr(st.session_state, 'tokens_loaded'):
            # Load Withings tokens
            withings_tokens = token_storage.load_tokens('withings')
            if withings_tokens:
                st.session_state.withings_access_token = withings_tokens['access_token']
                st.session_state.withings_refresh_token = withings_tokens.get('refresh_token')
            
            # Load Strava tokens
            strava_tokens = token_storage.load_tokens('strava')
            if strava_tokens:
                st.session_state.access_token = strava_tokens['access_token']
                st.session_state.refresh_token = strava_tokens.get('refresh_token')
                st.session_state.athlete_id = strava_tokens.get('athlete_id')
            
            st.session_state.tokens_loaded = True
        
        # Handle OAuth callbacks with better detection
        if 'code' in query_params:
            # Determine which service this callback is for
            if 'state' in query_params and 'scope' not in query_params:
                # Withings callback (has state but no scope)
                try:
                    code = query_params['code']
                    state = query_params['state']
                    withings_token_data = withings_auth.exchange_code_for_token(code, state)
                    
                    # Save to both session state and file
                    st.session_state.withings_access_token = withings_token_data['access_token']
                    st.session_state.withings_refresh_token = withings_token_data.get('refresh_token')
                    
                    # Save to persistent storage
                    token_storage.save_tokens('withings', withings_token_data)
                    
                    # Clear URL parameters and show success
                    st.query_params.clear()
                    st.success("Successfully connected to Withings!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Withings authentication failed: {str(e)}")
                    if "Rate limited" in str(e):
                        st.info("Please wait a moment before trying to connect to Withings again.")
            
            elif 'scope' in query_params:
                # Strava callback (has scope parameter)
                try:
                    code = query_params['code']
                    token_data = auth.exchange_code_for_token(code)
                    
                    # Save to both session state and file
                    st.session_state.access_token = token_data['access_token']
                    st.session_state.refresh_token = token_data.get('refresh_token')
                    st.session_state.athlete_id = token_data['athlete']['id']
                    
                    # Save to persistent storage
                    token_storage.save_tokens('strava', token_data)
                    
                    # Clear URL parameters and show success
                    st.query_params.clear()
                    st.success("Successfully authenticated with Strava!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Strava authentication failed: {str(e)}")
            
            else:
                # Unknown callback - clear it
                st.query_params.clear()
                st.warning("Unknown authentication callback received. Please try connecting again.")
        
        # Withings Authentication (Priority)
        st.subheader("🏥 Withings Health Data")
        if not withings_auth.is_authenticated():
            st.write("Connect your Withings account first to analyze your health data.")
            
            if st.button("Connect to Withings"):
                withings_auth_url = withings_auth.get_authorization_url()
                st.markdown(f'<a href="{withings_auth_url}" target="_self">Click here to authorize Withings</a>', unsafe_allow_html=True)
        else:
            st.success("✅ Connected to Withings")
            if st.button("Disconnect Withings"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key.startswith('withings_'):
                        del st.session_state[key]
                # Clear persistent storage
                token_storage.clear_tokens('withings')
                st.rerun()
        
        # Strava Authentication (Secondary)
        st.subheader("🏃 Strava Exercise Data")
        if not auth.is_authenticated():
            st.write("Optional: Connect Strava for exercise correlation analysis")
            
            if st.button("Connect to Strava"):
                auth_url = auth.get_authorization_url()
                st.markdown(f'<a href="{auth_url}" target="_self">Click here to authorize Strava</a>', unsafe_allow_html=True)
        else:
            st.success("✅ Connected to Strava")
            if st.button("Disconnect Strava"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key.startswith('access_token') or key.startswith('refresh_token') or key.startswith('athlete_'):
                        del st.session_state[key]
                # Clear persistent storage
                token_storage.clear_tokens('strava')
                st.rerun()
        
        # Navigation menu
        if withings_auth.is_authenticated():
            page = st.selectbox(
                "Choose a page:",
                ["Health Dashboard", "Weight Tracking", "Sleep Analysis", "Body Composition", "Exercise Analysis", "Correlation Analysis"]
            )
        else:
            page = "Welcome"
        
        # Clear authentication button for debugging
        if st.button("🔄 Clear All & Restart"):
            st.query_params.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Clear all persistent storage
            token_storage.clear_tokens()
            st.rerun()
    
    # Main content area
    if withings_auth.is_authenticated():
        withings_api = WithingsAPI(withings_auth.get_access_token())
        
        if page == "Health Dashboard":
            show_health_dashboard(withings_api)
        elif page == "Weight Tracking":
            show_weight_tracking(withings_api)
        elif page == "Sleep Analysis":
            show_sleep_analysis(withings_api)
        elif page == "Body Composition":
            show_body_composition(withings_api)
        elif page == "Exercise Analysis":
            if auth.is_authenticated():
                strava_api = StravaAPI(auth.get_access_token())
                show_exercise_analysis(strava_api)
            else:
                st.warning("Connect Strava to view exercise analysis")
                st.info("You can still view health data from other pages without Strava.")
        elif page == "Correlation Analysis":
            if auth.is_authenticated():
                strava_api = StravaAPI(auth.get_access_token())
                show_correlation_analysis(strava_api)
            else:
                st.warning("Connect Strava to view correlation analysis")
                st.info("Correlation analysis requires both Withings and Strava data.")
    
    else:
        # Welcome page for unauthenticated users
        st.markdown("""
        ## Welcome to Health Data Analyzer! 🏥
        
        This application helps you analyze your health and fitness data from Withings and Strava.
        
        ### 🏥 Withings Features:
        - ⚖️ **Weight Tracking**: Monitor weight trends over time
        - 😴 **Sleep Analysis**: Deep sleep, REM sleep, and sleep quality
        - 🫀 **Body Composition**: Fat %, muscle mass, BMI, and more
        - 📊 **Health Dashboard**: Comprehensive overview of your health metrics
        
        ### 🏃 Strava Features (Optional):
        - 📊 **Exercise Analytics**: Distance, pace, elevation trends
        - 🏆 **Performance Metrics**: Personal records and training zones
        - 📈 **Seasonal Analysis**: Year-over-year exercise patterns
        - 🔗 **Correlation Analysis**: How exercise affects sleep and weight
        
        ### Getting Started:
        1. **Connect to Withings** first (required for health data)
        2. **Connect to Strava** (optional, for exercise correlation)
        3. **Explore your health journey!**
        
        ---
        
        **Priority**: Start with Withings connection to see your health data immediately.
        """)

def show_health_dashboard(withings_api):
    st.header("🏥 Health Dashboard")
    
    try:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # Fetch all Withings data
        with st.spinner("Fetching health data..."):
            weight_data = withings_api.get_weight_measurements(start_datetime, end_datetime)
            weight_df = withings_api.weight_to_dataframe(weight_data)
            
            sleep_data = withings_api.get_sleep_data(start_datetime, end_datetime)
            sleep_df = withings_api.sleep_to_dataframe(sleep_data)
        
        # Key health metrics
        st.subheader("📊 Key Health Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not weight_df.empty and 'weight' in weight_df.columns:
                latest_weight = weight_df['weight'].iloc[-1]
                st.metric("Current Weight", f"{latest_weight:.1f} kg")
            else:
                st.metric("Weight", "No data")
        
        with col2:
            if not weight_df.empty and 'bmi' in weight_df.columns:
                latest_bmi = weight_df['bmi'].iloc[-1]
                st.metric("BMI", f"{latest_bmi:.1f}")
            else:
                st.metric("BMI", "No data")
        
        with col3:
            if not sleep_df.empty:
                avg_sleep = sleep_df['sleep_duration'].mean()
                st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs")
            else:
                st.metric("Sleep", "No data")
        
        with col4:
            if not sleep_df.empty and 'sleep_score' in sleep_df.columns:
                avg_score = sleep_df['sleep_score'].mean()
                st.metric("Sleep Score", f"{avg_score:.0f}")
            else:
                st.metric("Sleep Score", "No data")
        
        # Weight trend
        if not weight_df.empty:
            st.subheader("⚖️ Weight Trend")
            fig_weight = px.line(
                weight_df, 
                x='date', 
                y='weight',
                title='Weight Over Time',
                labels={'weight': 'Weight (kg)', 'date': 'Date'}
            )
            st.plotly_chart(fig_weight, use_container_width=True)
        
        # Sleep overview
        if not sleep_df.empty:
            st.subheader("😴 Sleep Overview")
            fig_sleep = px.line(
                sleep_df,
                x='date',
                y='sleep_duration',
                title='Sleep Duration Over Time',
                labels={'sleep_duration': 'Sleep Duration (hours)', 'date': 'Date'}
            )
            st.plotly_chart(fig_sleep, use_container_width=True)
        
        if weight_df.empty and sleep_df.empty:
            st.info("No health data found for the selected date range. Try expanding the date range or check your Withings data.")
        
    except Exception as e:
        st.error(f"Error loading health dashboard: {str(e)}")

def show_weight_tracking(withings_api):
    st.header("⚖️ Weight Tracking")
    
    try:
        # Extended date range for weight tracking
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90), key="weight_start")
        with col2:
            end_date = st.date_input("End Date", datetime.now(), key="weight_end")
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        with st.spinner("Fetching weight data..."):
            weight_data = withings_api.get_weight_measurements(start_datetime, end_datetime)
            df = withings_api.weight_to_dataframe(weight_data)
        
        if df.empty:
            st.warning("No weight data found for the selected date range.")
            return
        
        # Weight statistics
        st.subheader("📊 Weight Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_weight = df['weight'].iloc[-1]
            st.metric("Current Weight", f"{current_weight:.1f} kg")
        
        with col2:
            if len(df) > 1:
                weight_change = df['weight'].iloc[-1] - df['weight'].iloc[0]
                st.metric("Weight Change", f"{weight_change:+.1f} kg")
            else:
                st.metric("Weight Change", "N/A")
        
        with col3:
            min_weight = df['weight'].min()
            st.metric("Minimum Weight", f"{min_weight:.1f} kg")
        
        with col4:
            max_weight = df['weight'].max()
            st.metric("Maximum Weight", f"{max_weight:.1f} kg")
        
        # Weight trend with moving average
        st.subheader("📈 Weight Trend Analysis")
        df['weight_ma7'] = df['weight'].rolling(window=7, center=True).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['weight'],
            mode='markers',
            name='Daily Weight',
            marker=dict(color='lightblue')
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['weight_ma7'],
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title='Weight Tracking with Moving Average',
            xaxis_title='Date',
            yaxis_title='Weight (kg)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export weight data
        st.subheader("📄 Export Data")
        if st.button("Export Weight Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Weight CSV",
                data=csv,
                file_name=f"weight_data_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error loading weight tracking: {str(e)}")

def show_sleep_analysis(withings_api):
    st.header("😴 Sleep Analysis")
    
    try:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30), key="sleep_start")
        with col2:
            end_date = st.date_input("End Date", datetime.now(), key="sleep_end")
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        with st.spinner("Fetching sleep data..."):
            sleep_data = withings_api.get_sleep_data(start_datetime, end_datetime)
            df = withings_api.sleep_to_dataframe(sleep_data)
        
        if df.empty:
            st.warning("No sleep data found for the selected date range.")
            return
        
        # Sleep statistics
        st.subheader("📊 Sleep Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sleep = df['sleep_duration'].mean()
            st.metric("Average Sleep", f"{avg_sleep:.1f} hrs")
        
        with col2:
            if 'deep_sleep' in df.columns:
                avg_deep = df['deep_sleep'].mean()
                st.metric("Average Deep Sleep", f"{avg_deep:.1f} hrs")
            else:
                st.metric("Deep Sleep", "No data")
        
        with col3:
            if 'rem_sleep' in df.columns:
                avg_rem = df['rem_sleep'].mean()
                st.metric("Average REM Sleep", f"{avg_rem:.1f} hrs")
            else:
                st.metric("REM Sleep", "No data")
        
        with col4:
            if 'sleep_score' in df.columns:
                avg_score = df['sleep_score'].mean()
                st.metric("Average Sleep Score", f"{avg_score:.0f}")
            else:
                st.metric("Sleep Score", "No data")
        
        # Sleep duration trend
        st.subheader("🌙 Sleep Duration Trend")
        fig_duration = px.line(
            df,
            x='date',
            y='sleep_duration',
            title='Sleep Duration Over Time',
            labels={'sleep_duration': 'Sleep Duration (hours)', 'date': 'Date'}
        )
        fig_duration.add_hline(y=8, line_dash="dash", line_color="green", 
                              annotation_text="Recommended 8 hours")
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Sleep stages breakdown
        if 'deep_sleep' in df.columns and 'light_sleep' in df.columns and 'rem_sleep' in df.columns:
            st.subheader("🧠 Sleep Stages Analysis")
            
            # Average sleep stages
            avg_stages = {
                'Deep Sleep': df['deep_sleep'].mean(),
                'Light Sleep': df['light_sleep'].mean(),
                'REM Sleep': df['rem_sleep'].mean()
            }
            
            fig_pie = px.pie(
                values=list(avg_stages.values()),
                names=list(avg_stages.keys()),
                title='Average Sleep Stages Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Sleep quality over time
        if 'sleep_score' in df.columns:
            st.subheader("💤 Sleep Quality Score")
            fig_score = px.line(
                df,
                x='date',
                y='sleep_score',
                title='Sleep Quality Score Over Time',
                labels={'sleep_score': 'Sleep Score', 'date': 'Date'}
            )
            fig_score.add_hline(y=80, line_dash="dash", line_color="green", 
                               annotation_text="Good Sleep (80+)")
            st.plotly_chart(fig_score, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading sleep analysis: {str(e)}")

def show_body_composition(withings_api):
    st.header("🫀 Body Composition Analysis")
    
    try:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90), key="body_start")
        with col2:
            end_date = st.date_input("End Date", datetime.now(), key="body_end")
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        with st.spinner("Fetching body composition data..."):
            weight_data = withings_api.get_weight_measurements(start_datetime, end_datetime)
            df = withings_api.weight_to_dataframe(weight_data)
        
        if df.empty:
            st.warning("No body composition data found for the selected date range.")
            return
        
        # Current body composition metrics
        st.subheader("📊 Current Body Composition")
        latest_data = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'fat_percent' in df.columns:
                st.metric("Body Fat %", f"{latest_data.get('fat_percent', 0):.1f}%")
            else:
                st.metric("Body Fat %", "No data")
        
        with col2:
            if 'muscle_percent' in df.columns:
                st.metric("Muscle %", f"{latest_data.get('muscle_percent', 0):.1f}%")
            else:
                st.metric("Muscle %", "No data")
        
        with col3:
            if 'bmi' in df.columns:
                st.metric("BMI", f"{latest_data.get('bmi', 0):.1f}")
            else:
                st.metric("BMI", "No data")
        
        with col4:
            if 'muscle_mass' in df.columns:
                st.metric("Muscle Mass", f"{latest_data.get('muscle_mass', 0):.1f} kg")
            else:
                st.metric("Muscle Mass", "No data")
        
        # Body composition trends
        composition_cols = ['fat_percent', 'muscle_percent', 'bone_percent']
        available_cols = [col for col in composition_cols if col in df.columns and df[col].notna().any()]
        
        if available_cols:
            st.subheader("📈 Body Composition Trends")
            fig = go.Figure()
            
            colors = ['red', 'blue', 'green']
            for i, col in enumerate(available_cols):
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title='Body Composition Over Time',
                xaxis_title='Date',
                yaxis_title='Percentage (%)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # BMI trend and categories
        if 'bmi' in df.columns:
            st.subheader("📏 BMI Analysis")
            
            fig_bmi = px.line(
                df,
                x='date',
                y='bmi',
                title='BMI Trend Over Time',
                labels={'bmi': 'BMI', 'date': 'Date'}
            )
            
            # Add BMI category lines
            fig_bmi.add_hline(y=18.5, line_dash="dash", line_color="blue", 
                             annotation_text="Underweight")
            fig_bmi.add_hline(y=25, line_dash="dash", line_color="orange", 
                             annotation_text="Overweight")
            fig_bmi.add_hline(y=30, line_dash="dash", line_color="red", 
                             annotation_text="Obese")
            
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Body mass trends
        mass_cols = ['fat_mass', 'muscle_mass', 'bone_mass']
        available_mass_cols = [col for col in mass_cols if col in df.columns and df[col].notna().any()]
        
        if available_mass_cols:
            st.subheader("⚖️ Body Mass Composition")
            fig_mass = go.Figure()
            
            for col in available_mass_cols:
                fig_mass.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title()
                ))
            
            fig_mass.update_layout(
                title='Body Mass Components Over Time',
                xaxis_title='Date',
                yaxis_title='Mass (kg)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_mass, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading body composition: {str(e)}")

def show_exercise_analysis(strava_api):
    st.header("🏃 Exercise Analysis")
    st.info("This page shows your Strava exercise data. For correlation with health data, visit the Correlation Analysis page.")
    show_dashboard(strava_api)  # Reuse existing Strava dashboard

def show_dashboard(api):
    st.header("📊 Dashboard")
    
    try:
        # Get recent activities
        activities = api.get_activities(per_page=50)
        
        if not activities:
            st.warning("No activities found. Start recording some workouts!")
            return
        
        df = api.activities_to_dataframe(activities)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_activities = len(df)
            st.metric("Total Activities", total_activities)
        
        with col2:
            total_distance = df['distance_km'].sum()
            st.metric("Total Distance", f"{total_distance:.1f} km")
        
        with col3:
            total_time = df['moving_time_hours'].sum()
            st.metric("Total Time", f"{total_time:.1f} hours")
        
        with col4:
            avg_distance = df['distance_km'].mean()
            st.metric("Avg Distance", f"{avg_distance:.1f} km")
        
        # Get more historical data for comprehensive analysis
        all_activities = []
        page = 1
        while len(all_activities) < 1000:  # Limit to prevent too many API calls
            batch = api.get_activities(page=page, per_page=100)
            if not batch:
                break
            all_activities.extend(batch)
            page += 1
            if len(batch) < 100:  # Last page
                break
        
        if all_activities:
            all_df = api.activities_to_dataframe(all_activities)
        else:
            all_df = df
        
        # Yearly and seasonal analysis
        st.subheader("📈 Long-term Exercise Trends & Seasonal Analysis")
        
        # Add time-based columns
        all_df['year'] = all_df['start_date_local'].dt.year
        all_df['month'] = all_df['start_date_local'].dt.month
        all_df['week_of_year'] = all_df['start_date_local'].dt.isocalendar().week
        all_df['season'] = all_df['month'].map({
            1: 'Winter', 2: 'Winter', 3: 'Spring',
            4: 'Spring', 5: 'Spring', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Fall',
            10: 'Fall', 11: 'Fall', 12: 'Winter'
        })
        
        # Create comprehensive time series
        all_df['year_week'] = all_df['start_date_local'].dt.to_period('W')
        
        # Weekly aggregations for multi-year view
        weekly_comprehensive = all_df.groupby('year_week').agg({
            'distance_km': 'sum',
            'moving_time_hours': 'sum',
            'id': 'count'
        }).reset_index()
        weekly_comprehensive['date'] = weekly_comprehensive['year_week'].dt.start_time
        weekly_comprehensive['year'] = weekly_comprehensive['date'].dt.year
        weekly_comprehensive['week_num'] = weekly_comprehensive['date'].dt.isocalendar().week
        
        # Multi-year weekly exercise volume
        col1, col2 = st.columns(2)
        
        with col1:
            fig_weekly_time = px.line(
                weekly_comprehensive,
                x='date',
                y='moving_time_hours',
                title='Weekly Exercise Time Over Years',
                labels={'moving_time_hours': 'Hours per Week', 'date': 'Date'}
            )
            fig_weekly_time.add_scatter(
                x=weekly_comprehensive['date'],
                y=weekly_comprehensive['moving_time_hours'].rolling(window=4, center=True).mean(),
                mode='lines',
                name='4-week Moving Average',
                line=dict(color='red', width=2)
            )
            st.plotly_chart(fig_weekly_time, use_container_width=True)
        
        with col2:
            fig_weekly_distance = px.line(
                weekly_comprehensive,
                x='date',
                y='distance_km',
                title='Weekly Distance Over Years',
                labels={'distance_km': 'Distance (km) per Week', 'date': 'Date'}
            )
            fig_weekly_distance.add_scatter(
                x=weekly_comprehensive['date'],
                y=weekly_comprehensive['distance_km'].rolling(window=4, center=True).mean(),
                mode='lines',
                name='4-week Moving Average',
                line=dict(color='red', width=2)
            )
            st.plotly_chart(fig_weekly_distance, use_container_width=True)
        
        # Seasonal Analysis
        st.subheader("🌟 Seasonal Patterns")
        
        # Seasonal aggregations
        seasonal_stats = all_df.groupby(['year', 'season']).agg({
            'distance_km': 'sum',
            'moving_time_hours': 'sum',
            'id': 'count'
        }).reset_index()
        
        # Average by season across all years
        season_avg = all_df.groupby('season').agg({
            'distance_km': 'mean',
            'moving_time_hours': 'mean',
            'id': 'count'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_seasonal_time = px.bar(
                season_avg,
                x='season',
                y='moving_time_hours',
                title='Average Exercise Time by Season',
                labels={'moving_time_hours': 'Avg Hours per Activity', 'season': 'Season'},
                color='season',
                category_orders={'season': ['Spring', 'Summer', 'Fall', 'Winter']}
            )
            st.plotly_chart(fig_seasonal_time, use_container_width=True)
        
        with col2:
            fig_seasonal_distance = px.bar(
                season_avg,
                x='season',
                y='distance_km',
                title='Average Distance by Season',
                labels={'distance_km': 'Avg Distance (km) per Activity', 'season': 'Season'},
                color='season',
                category_orders={'season': ['Spring', 'Summer', 'Fall', 'Winter']}
            )
            st.plotly_chart(fig_seasonal_distance, use_container_width=True)
        
        # Yearly trends
        st.subheader("📊 Year-over-Year Trends")
        
        yearly_stats = all_df.groupby('year').agg({
            'distance_km': 'sum',
            'moving_time_hours': 'sum',
            'id': 'count'
        }).reset_index()
        
        if len(yearly_stats) > 1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_yearly_total = px.bar(
                    yearly_stats,
                    x='year',
                    y='moving_time_hours',
                    title='Total Exercise Hours by Year',
                    labels={'moving_time_hours': 'Total Hours', 'year': 'Year'}
                )
                st.plotly_chart(fig_yearly_total, use_container_width=True)
            
            with col2:
                fig_yearly_distance = px.bar(
                    yearly_stats,
                    x='year',
                    y='distance_km',
                    title='Total Distance by Year',
                    labels={'distance_km': 'Total Distance (km)', 'year': 'Year'}
                )
                st.plotly_chart(fig_yearly_distance, use_container_width=True)
            
            with col3:
                fig_yearly_activities = px.bar(
                    yearly_stats,
                    x='year',
                    y='id',
                    title='Total Activities by Year',
                    labels={'id': 'Number of Activities', 'year': 'Year'}
                )
                st.plotly_chart(fig_yearly_activities, use_container_width=True)
        
        # Heatmap of weekly patterns
        st.subheader("🔥 Exercise Intensity Heatmap")
        
        # Create a heatmap showing exercise volume by week of year across years
        if len(weekly_comprehensive) > 52:
            # Pivot data for heatmap
            heatmap_data = weekly_comprehensive.pivot_table(
                index='year',
                columns='week_num',
                values='moving_time_hours',
                fill_value=0
            )
            
            fig_heatmap = px.imshow(
                heatmap_data,
                title='Exercise Hours by Week Across Years',
                labels={'x': 'Week of Year', 'y': 'Year', 'color': 'Hours'},
                aspect='auto'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Activity type distribution
        st.subheader("🏃 Activity Types")
        activity_counts = all_df['type'].value_counts()
        
        fig_pie = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            title='Activity Distribution (All Time)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def show_activities(api):
    st.header("🏃 Activities")
    
    try:
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Get activities
        activities = api.get_activities(
            after=datetime.combine(start_date, datetime.min.time()),
            before=datetime.combine(end_date, datetime.min.time()),
            per_page=100
        )
        
        if not activities:
            st.warning("No activities found for the selected date range.")
            return
        
        df = api.activities_to_dataframe(activities)
        
        # Activity table
        st.subheader("Recent Activities")
        
        # Select columns to display
        display_columns = [
            'name', 'type', 'start_date_local', 'distance_km', 
            'moving_time_hours', 'total_elevation_gain'
        ]
        
        display_df = df[display_columns].copy()
        display_df['start_date_local'] = display_df['start_date_local'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.round(2)
        
        st.dataframe(
            display_df,
            column_config={
                "name": "Activity Name",
                "type": "Type",
                "start_date_local": "Date",
                "distance_km": "Distance (km)",
                "moving_time_hours": "Time (hours)",
                "total_elevation_gain": "Elevation (m)"
            },
            use_container_width=True
        )
        
        # Export functionality
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"strava_activities_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error loading activities: {str(e)}")

def show_performance_analytics(api):
    st.header("🏆 Performance Analytics")
    st.info("Performance analytics coming soon! This will include personal records, training zones, and improvement trends.")

def show_profile(api):
    st.header("👤 Athlete Profile")
    
    try:
        athlete = api.get_athlete()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Info")
            st.write(f"**Name:** {athlete.get('firstname', '')} {athlete.get('lastname', '')}")
            st.write(f"**Location:** {athlete.get('city', 'Not specified')}, {athlete.get('country', '')}")
            st.write(f"**Member Since:** {athlete.get('created_at', 'Unknown')[:10]}")
            
        with col2:
            st.subheader("Stats")
            if 'follower_count' in athlete:
                st.write(f"**Followers:** {athlete['follower_count']}")
            if 'friend_count' in athlete:
                st.write(f"**Following:** {athlete['friend_count']}")
        
        # Get athlete stats if available
        try:
            stats = api.get_athlete_stats(athlete['id'])
            
            st.subheader("All-Time Stats")
            
            col1, col2, col3 = st.columns(3)
            
            if 'all_run_totals' in stats:
                with col1:
                    st.metric(
                        "Total Run Distance", 
                        f"{stats['all_run_totals']['distance'] / 1000:.0f} km"
                    )
                    st.metric(
                        "Total Run Time", 
                        f"{stats['all_run_totals']['moving_time'] / 3600:.0f} hours"
                    )
            
            if 'all_ride_totals' in stats:
                with col2:
                    st.metric(
                        "Total Ride Distance", 
                        f"{stats['all_ride_totals']['distance'] / 1000:.0f} km"
                    )
                    st.metric(
                        "Total Ride Time", 
                        f"{stats['all_ride_totals']['moving_time'] / 3600:.0f} hours"
                    )
            
        except Exception as e:
            st.warning("Could not load detailed stats.")
        
    except Exception as e:
        st.error(f"Error loading profile: {str(e)}")

def show_correlation_analysis(strava_api):
    st.header("📈 Exercise, Sleep & Weight Correlation Analysis")
    
    if not withings_auth.is_authenticated():
        st.warning("Please connect your Withings account to analyze correlations with weight and sleep data.")
        return
    
    try:
        withings_api = WithingsAPI(withings_auth.get_access_token())
        analyzer = CorrelationAnalysis()
        exercise_sleep_analyzer = ExerciseSleepAnalysis()
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # Fetch data
        with st.spinner("Fetching data from Strava and Withings..."):
            # Strava data
            strava_activities = strava_api.get_activities(
                after=start_datetime,
                before=end_datetime,
                per_page=200
            )
            strava_df = strava_api.activities_to_dataframe(strava_activities) if strava_activities else pd.DataFrame()
            
            # Withings data
            weight_data = withings_api.get_weight_measurements(start_datetime, end_datetime)
            weight_df = withings_api.weight_to_dataframe(weight_data)
            
            sleep_data = withings_api.get_sleep_data(start_datetime, end_datetime)
            sleep_df = withings_api.sleep_to_dataframe(sleep_data)
        
        # ===============================
        # NEW: Advanced Exercise-Sleep Analysis
        # ===============================
        
        st.subheader("🎯 Exercise Impact on Sleep Quality (Advanced Analysis)")
        st.info("This analysis examines how exercise timing and intensity affect your sleep over the next 1-2 nights, using machine learning models.")
        
        if not strava_df.empty and not sleep_df.empty:
            # Prepare exercise-sleep impact data
            exercise_sleep_df = exercise_sleep_analyzer.prepare_exercise_sleep_data(strava_df, sleep_df)
            
            if not exercise_sleep_df.empty and len(exercise_sleep_df) > 5:
                # Perform regression analysis
                regression_results = exercise_sleep_analyzer.perform_regression_analysis(exercise_sleep_df)
                
                # Analyze timing effects
                timing_analysis = exercise_sleep_analyzer.analyze_exercise_timing_effects(exercise_sleep_df)
                
                # Generate insights
                st.subheader("🧠 Key Insights: Exercise → Sleep Impact")
                insights = exercise_sleep_analyzer.generate_insights(exercise_sleep_df, regression_results, timing_analysis)
                for insight in insights:
                    st.markdown(insight)
                
                # Exercise timing visualization
                timing_plot = exercise_sleep_analyzer.create_exercise_timing_plot(exercise_sleep_df)
                if timing_plot:
                    st.plotly_chart(timing_plot, use_container_width=True)
                
                # Scatter plots for exercise-sleep relationships
                scatter_plots = exercise_sleep_analyzer.create_exercise_sleep_scatter(exercise_sleep_df)
                for plot in scatter_plots:
                    st.plotly_chart(plot, use_container_width=True)
                
                # Model performance summary
                if regression_results:
                    st.subheader("🔬 Model Performance Summary")
                    
                    model_data = []
                    for target, results in regression_results.items():
                        model_data.append({
                            'Sleep Metric': target.replace('night1_', 'Night 1: ').replace('night2_', 'Night 2: ').replace('_', ' ').title(),
                            'Linear R²': f"{results['linear_r2']:.3f}",
                            'Random Forest R²': f"{results['rf_r2']:.3f}",
                            'Sample Size': results['sample_size'],
                            'Top Factor': max(results['feature_importance'].keys(), key=lambda x: results['feature_importance'][x]).replace('_', ' ').title()
                        })
                    
                    model_df = pd.DataFrame(model_data)
                    st.dataframe(model_df, use_container_width=True)
                
                # Export exercise-sleep analysis
                st.subheader("📄 Export Exercise-Sleep Analysis")
                if st.button("Export Exercise-Sleep Data"):
                    csv = exercise_sleep_df.to_csv(index=False)
                    st.download_button(
                        label="Download Exercise-Sleep Analysis CSV",
                        data=csv,
                        file_name=f"exercise_sleep_analysis_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Insufficient data for advanced exercise-sleep analysis. Need at least 5 exercise sessions with corresponding sleep data.")
        else:
            st.warning("Need both exercise and sleep data for advanced analysis.")
        
        # ===============================
        # EXISTING: General Correlations
        # ===============================
        
        st.subheader("📊 General Health Correlations")
        
        # Combine data
        combined_df = analyzer.prepare_combined_data(strava_df, weight_df, sleep_df)
        
        if combined_df.empty:
            st.warning("No overlapping data found between Strava and Withings for the selected date range.")
            return
        
        # Display data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exercise_days = combined_df['daily_exercise_time'].notna().sum() if 'daily_exercise_time' in combined_df.columns else 0
            st.metric("Exercise Days", exercise_days)
        
        with col2:
            weight_measurements = combined_df['weight'].notna().sum() if 'weight' in combined_df.columns else 0
            st.metric("Weight Measurements", weight_measurements)
        
        with col3:
            sleep_days = combined_df['sleep_duration'].notna().sum() if 'sleep_duration' in combined_df.columns else 0
            st.metric("Sleep Records", sleep_days)
        
        # Calculate correlations
        correlations = analyzer.calculate_correlations(combined_df)
        
        # Display correlation insights
        insights = analyzer.generate_insights(correlations, combined_df)
        for insight in insights:
            st.write(insight)
        
        # Correlation matrix
        correlation_matrix_fig = analyzer.create_correlation_matrix(combined_df)
        if correlation_matrix_fig:
            st.plotly_chart(correlation_matrix_fig, use_container_width=True)
        else:
            st.info("Not enough data for correlation matrix visualization.")
        
        # Time series comparison
        time_series_fig = analyzer.create_time_series_comparison(combined_df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
        else:
            st.info("Not enough data for time series comparison.")
        
        # Scatter plots
        scatter_plots = analyzer.create_scatter_plots(combined_df)
        
        if scatter_plots:
            for i, fig in enumerate(scatter_plots):
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot analysis.")
        
        # Raw data display (optional)
        with st.expander("🔍 View Raw Combined Data"):
            st.dataframe(combined_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
        st.error("Make sure both Strava and Withings are properly connected.")

if __name__ == "__main__":
    main()