import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

class WithingsAPI:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = 'https://wbsapi.withings.net'
        
    def make_request(self, endpoint, params=None):
        """Make authenticated request to Withings API"""
        if params is None:
            params = {}
        
        params['access_token'] = self.access_token
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 0:  # Success
                return data.get('body', {})
            else:
                raise Exception(f"API error: {data.get('error', 'Unknown error')}")
        else:
            raise Exception(f"HTTP error: {response.status_code} - {response.text}")
    
    def get_user_info(self):
        """Get user profile information"""
        return self.make_request('/v2/user', {'action': 'getdevice'})
    
    def get_weight_measurements(self, start_date=None, end_date=None):
        """Get weight and body composition measurements"""
        params = {
            'action': 'getmeas',
            'category': 1,  # Real measurements only
            'meastypes': '1,5,6,8,76,77,88,91,123'  # Weight, fat %, muscle %, bone %, BMI, fat mass, muscle mass, bone mass, metabolic age
        }
        
        if start_date:
            params['startdate'] = int(start_date.timestamp())
        if end_date:
            params['enddate'] = int(end_date.timestamp())
        
        try:
            data = self.make_request('/measure', params)
            return data.get('measuregrps', [])
        except Exception as e:
            st.warning(f"Could not fetch weight data: {str(e)}")
            return []
    
    def get_sleep_data(self, start_date=None, end_date=None):
        """Get sleep measurements"""
        # Ensure we have date parameters (Withings requires them)
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Try the sleep summary endpoint first (more reliable)
        params = {
            'action': 'getsummary',
            'startdateymd': start_date.strftime('%Y-%m-%d'),
            'enddateymd': end_date.strftime('%Y-%m-%d')
        }
        
        try:
            print(f"Sleep API params: {params}")  # Debug
            data = self.make_request('/v2/sleep', params)
            return data.get('series', [])
        except Exception as e:
            print(f"Sleep summary API error: {str(e)}")  # Debug
            
            # Try the detailed sleep endpoint
            try:
                detail_params = {
                    'action': 'get',
                    'startdateymd': start_date.strftime('%Y-%m-%d'),
                    'enddateymd': end_date.strftime('%Y-%m-%d')
                }
                
                print(f"Trying detailed sleep params: {detail_params}")  # Debug
                data = self.make_request('/v2/sleep', detail_params)
                return data.get('series', [])
            except Exception as e2:
                print(f"Detailed sleep API error: {str(e2)}")  # Debug
                
                # If both fail, try without date range (last 30 days default)
                try:
                    fallback_params = {'action': 'getsummary'}
                    print(f"Trying fallback sleep params: {fallback_params}")  # Debug
                    data = self.make_request('/v2/sleep', fallback_params)
                    return data.get('series', [])
                except Exception as e3:
                    print(f"Fallback sleep API error: {str(e3)}")  # Debug
                    st.warning("Could not fetch sleep data. This may be due to: no sleep tracking device connected, insufficient permissions, or no sleep data available for the selected period.")
                    return []
    
    def get_activity_data(self, start_date=None, end_date=None):
        """Get activity/steps data"""
        params = {
            'action': 'getmeas',
            'category': 1,
            'meastypes': '36,40'  # Steps, active calories
        }
        
        if start_date:
            params['startdate'] = int(start_date.timestamp())
        if end_date:
            params['enddate'] = int(end_date.timestamp())
        
        try:
            data = self.make_request('/measure', params)
            return data.get('measuregrps', [])
        except Exception as e:
            st.warning(f"Could not fetch activity data: {str(e)}")
            return []
    
    def weight_to_dataframe(self, weight_data):
        """Convert weight measurements to pandas DataFrame"""
        if not weight_data:
            return pd.DataFrame()
        
        records = []
        for group in weight_data:
            date = datetime.fromtimestamp(group['date'])
            record = {'date': date}
            
            for measure in group['measures']:
                measure_type = measure['type']
                value = measure['value'] * (10 ** measure['unit'])
                
                # Map measure types to readable names
                type_mapping = {
                    1: 'weight',
                    5: 'fat_percent',
                    6: 'muscle_percent', 
                    8: 'bone_percent',
                    76: 'muscle_mass',
                    77: 'bone_mass',
                    88: 'fat_mass',
                    91: 'bmi',
                    123: 'metabolic_age'
                }
                
                if measure_type in type_mapping:
                    record[type_mapping[measure_type]] = value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
        
        return df
    
    def sleep_to_dataframe(self, sleep_data):
        """Convert sleep data to pandas DataFrame"""
        if not sleep_data:
            return pd.DataFrame()
        
        records = []
        for sleep_session in sleep_data:
            date = datetime.fromtimestamp(sleep_session['startdate'])
            
            record = {
                'date': date,
                # total sleep time is provided in seconds; convert to hours
                'sleep_duration': (sleep_session.get('enddate', sleep_session['startdate']) - sleep_session['startdate']) / 3600,
                'deep_sleep': sleep_session.get('data', {}).get('deepsleepduration', 0) / 60,
                'light_sleep': sleep_session.get('data', {}).get('lightsleepduration', 0) / 60,
                'rem_sleep': sleep_session.get('data', {}).get('remsleepduration', 0) / 60,
                'wake_duration': sleep_session.get('data', {}).get('wakeupcount', 0),
                'sleep_score': sleep_session.get('data', {}).get('sleep_score', 0)
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
        
        return df
    
    def activity_to_dataframe(self, activity_data):
        """Convert activity data to pandas DataFrame"""
        if not activity_data:
            return pd.DataFrame()
        
        records = []
        for group in activity_data:
            date = datetime.fromtimestamp(group['date'])
            record = {'date': date}
            
            for measure in group['measures']:
                measure_type = measure['type']
                value = measure['value'] * (10 ** measure['unit'])
                
                if measure_type == 36:  # Steps
                    record['steps'] = value
                elif measure_type == 40:  # Active calories
                    record['active_calories'] = value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('date')
            df = df.reset_index(drop=True)
        
        return df

