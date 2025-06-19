import streamlit as st
import requests
import os
import hashlib
import base64
import secrets
from urllib.parse import urlencode, parse_qs, urlparse
from dotenv import load_dotenv

load_dotenv()

class WithingsAuth:
    def __init__(self):
        # Try Streamlit secrets first, then environment variables
        try:
            self.client_id = st.secrets.get('WITHINGS_CLIENT_ID') or os.getenv('WITHINGS_CLIENT_ID')
            self.client_secret = st.secrets.get('WITHINGS_CLIENT_SECRET') or os.getenv('WITHINGS_CLIENT_SECRET')
            self.redirect_uri = st.secrets.get('WITHINGS_REDIRECT_URI') or os.getenv('WITHINGS_REDIRECT_URI', 'http://localhost:8501')
        except:
            # Fallback to environment variables only
            self.client_id = os.getenv('WITHINGS_CLIENT_ID')
            self.client_secret = os.getenv('WITHINGS_CLIENT_SECRET')
            self.redirect_uri = os.getenv('WITHINGS_REDIRECT_URI', 'http://localhost:8501')
        self.auth_url = 'https://account.withings.com/oauth2_user/authorize2'
        self.token_url = 'https://wbsapi.withings.net/v2/oauth2'
        self.scope = 'user.metrics,user.activity,user.sleepevents'
    
    def generate_code_verifier(self):
        """Generate code verifier for PKCE"""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
        # Remove padding
        return code_verifier.rstrip('=')
    
    def generate_code_challenge(self, code_verifier):
        """Generate code challenge from verifier"""
        digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    def get_authorization_url(self):
        """Generate Withings OAuth authorization URL"""
        if not self.client_id:
            raise Exception("Withings Client ID not found in environment variables")
        
        # Simplified OAuth without PKCE
        state = secrets.token_urlsafe(32)
        
        # Store in session state for later use
        st.session_state.withings_state = state
        
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': self.scope,
            'state': state
        }
        
        # Debug info (remove in production)
        print(f"Auth URL params: {params}")
        
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_token(self, authorization_code, state):
        """Exchange authorization code for access token"""
        # Use the correct Withings token exchange format
        
        data = {
            'action': 'requesttoken',
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': authorization_code,
            'redirect_uri': self.redirect_uri
        }
        
        print(f"Token exchange data: {data}")  # Debug
        
        response = requests.post(self.token_url, data=data)
        
        print(f"Response status: {response.status_code}")  # Debug
        print(f"Response text: {response.text}")  # Debug
        
        if response.status_code == 200:
            token_data = response.json()
            if token_data.get('status') == 0 and 'body' in token_data:
                # Withings API returns data in 'body' field
                body = token_data['body']
                if 'access_token' in body:
                    return body
                else:
                    raise Exception(f"Token exchange failed: {token_data}")
            elif token_data.get('status') == 601:
                # Rate limit error
                wait_time = token_data.get('body', {}).get('wait_seconds', 10)
                raise Exception(f"Rate limited by Withings. Please wait {wait_time} seconds and try again.")
            else:
                raise Exception(f"Token exchange failed: {token_data}")
        else:
            raise Exception(f"Token exchange failed: {response.text}")
    
    def refresh_token(self, refresh_token):
        """Refresh an expired access token"""
        data = {
            'action': 'requesttoken',
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token
        }
        
        response = requests.post(self.token_url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            if token_data.get('status') == 0 and 'body' in token_data:
                body = token_data['body']
                if 'access_token' in body:
                    return body
                else:
                    raise Exception(f"Token refresh failed: {token_data}")
            else:
                raise Exception(f"Token refresh failed: {token_data}")
        else:
            raise Exception(f"Token refresh failed: {response.text}")
    
    def is_authenticated(self):
        """Check if user is authenticated with Withings"""
        return ('withings_access_token' in st.session_state and 
                st.session_state.withings_access_token is not None)
    
    def get_access_token(self):
        """Get current Withings access token"""
        return st.session_state.get('withings_access_token')