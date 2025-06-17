import streamlit as st
from withings_api import WithingsAPI
from datetime import datetime, timedelta

# Simple test script to debug Withings API calls
if 'withings_access_token' in st.session_state:
    api = WithingsAPI(st.session_state.withings_access_token)
    
    st.write("Testing Withings API calls...")
    
    # Test sleep data with different approaches
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    st.write(f"Testing sleep data from {start_date.date()} to {end_date.date()}")
    
    try:
        # Direct API call to see raw response
        import requests
        
        params = {
            'action': 'getsummary',
            'startdateymd': start_date.strftime('%Y-%m-%d'),
            'enddateymd': end_date.strftime('%Y-%m-%d'),
            'access_token': st.session_state.withings_access_token
        }
        
        response = requests.get('https://wbsapi.withings.net/v2/sleep', params=params)
        st.write("Raw sleep API response:")
        st.json(response.json())
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.write("Please connect to Withings first")