import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

class StravaAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = 'https://www.strava.com/api/v3'
        self.headers = {'Authorization': f'Bearer {access_token}'}
    
    def get_athlete(self):
        """Get authenticated athlete's profile"""
        url = f"{self.base_url}/athlete"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get athlete data: {response.text}")
    
    def get_activities(self, before=None, after=None, page=1, per_page=30):
        """Get athlete's activities"""
        url = f"{self.base_url}/athlete/activities"
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if before:
            params['before'] = int(before.timestamp())
        if after:
            params['after'] = int(after.timestamp())
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get activities: {response.text}")
    
    def get_activity_details(self, activity_id):
        """Get detailed information about a specific activity"""
        url = f"{self.base_url}/activities/{activity_id}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get activity details: {response.text}")
    
    def get_athlete_stats(self, athlete_id):
        """Get athlete's statistics"""
        url = f"{self.base_url}/athletes/{athlete_id}/stats"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get athlete stats: {response.text}")
    
    def activities_to_dataframe(self, activities):
        """Convert activities list to pandas DataFrame"""
        if not activities:
            return pd.DataFrame()
        
        df = pd.DataFrame(activities)
        
        # Convert datetime fields
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
        if 'start_date_local' in df.columns:
            df['start_date_local'] = pd.to_datetime(df['start_date_local'])
        
        # Convert distance from meters to kilometers
        if 'distance' in df.columns:
            df['distance_km'] = df['distance'] / 1000
        
        # Convert moving time to hours
        if 'moving_time' in df.columns:
            df['moving_time_hours'] = df['moving_time'] / 3600
        
        # Calculate pace (min/km) for running activities
        if 'distance' in df.columns and 'moving_time' in df.columns:
            df['pace_min_per_km'] = (df['moving_time'] / 60) / (df['distance'] / 1000)
        
        return df