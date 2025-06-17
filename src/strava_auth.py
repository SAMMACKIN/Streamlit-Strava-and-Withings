import streamlit as st
import requests
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

class StravaAuth:
    def __init__(self):
        self.client_id = os.getenv('STRAVA_CLIENT_ID')
        self.client_secret = os.getenv('STRAVA_CLIENT_SECRET')
        self.redirect_uri = os.getenv('STRAVA_REDIRECT_URI', 'http://localhost:8501')
        self.auth_url = 'https://www.strava.com/oauth/authorize'
        self.token_url = 'https://www.strava.com/oauth/token'
    
    def get_authorization_url(self):
        """Generate Strava OAuth authorization URL"""
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'approval_prompt': 'force',
            'scope': 'read,activity:read_all,profile:read_all'
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, authorization_code):
        """Exchange authorization code for access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(self.token_url, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Token exchange failed: {response.text}")
    
    def refresh_token(self, refresh_token):
        """Refresh an expired access token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        response = requests.post(self.token_url, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Token refresh failed: {response.text}")
    
    def is_authenticated(self):
        """Check if user is authenticated"""
        return 'access_token' in st.session_state and st.session_state.access_token is not None
    
    def get_access_token(self):
        """Get current access token"""
        return st.session_state.get('access_token')