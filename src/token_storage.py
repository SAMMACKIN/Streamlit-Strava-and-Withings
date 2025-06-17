import json
import os
from datetime import datetime

class TokenStorage:
    def __init__(self, storage_file='tokens.json'):
        self.storage_file = storage_file
    
    def save_tokens(self, service, tokens):
        """Save tokens to file"""
        try:
            # Load existing tokens
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    all_tokens = json.load(f)
            else:
                all_tokens = {}
            
            # Add timestamp
            tokens['saved_at'] = datetime.now().isoformat()
            
            # Save service tokens
            all_tokens[service] = tokens
            
            # Write back to file
            with open(self.storage_file, 'w') as f:
                json.dump(all_tokens, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving tokens: {e}")
            return False
    
    def load_tokens(self, service):
        """Load tokens from file"""
        try:
            if not os.path.exists(self.storage_file):
                return None
            
            with open(self.storage_file, 'r') as f:
                all_tokens = json.load(f)
            
            return all_tokens.get(service)
        except Exception as e:
            print(f"Error loading tokens: {e}")
            return None
    
    def clear_tokens(self, service=None):
        """Clear tokens for a service or all tokens"""
        try:
            if not os.path.exists(self.storage_file):
                return True
            
            if service is None:
                # Clear all tokens
                os.remove(self.storage_file)
                return True
            
            # Clear specific service
            with open(self.storage_file, 'r') as f:
                all_tokens = json.load(f)
            
            if service in all_tokens:
                del all_tokens[service]
            
            with open(self.storage_file, 'w') as f:
                json.dump(all_tokens, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error clearing tokens: {e}")
            return False
    
    def is_authenticated(self, service):
        """Check if service is authenticated"""
        tokens = self.load_tokens(service)
        return tokens is not None and 'access_token' in tokens
    
    def get_access_token(self, service):
        """Get access token for service"""
        tokens = self.load_tokens(service)
        if tokens:
            return tokens.get('access_token')
        return None