# dontknwo/firebase_utils.py
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime
import os

# Collection name for portfolio value history
PORTFOLIO_HISTORY_COLLECTION = "portfolio_value_history"
USER_ID_PLACEHOLDER = "default_user_portfolio" # Ensure this is consistent

@st.cache_resource
def initialize_firebase_admin():
    if not firebase_admin._apps:
        try:
            required_keys = [
                "type", "project_id", "private_key_id", "private_key",
                "client_email", "client_id", "auth_uri", "token_uri",
                "auth_provider_x509_cert_url", "client_x509_cert_url"
            ]
            secret_creds = st.secrets.get("firebase_service_account", {})
            if not all(key in secret_creds for key in required_keys):
                missing_keys = [key for key in required_keys if key not in secret_creds]
                st.error(f"Firebase secrets missing required keys: {', '.join(missing_keys)}. Cannot initialize from secrets.")
                raise KeyError(f"Missing Firebase secrets: {', '.join(missing_keys)}")

            cred_dict = {key: secret_creds[key] for key in required_keys}
            cred_dict["private_key"] = secret_creds["private_key"].replace('\\n', '\n')
            
            if "universe_domain" in secret_creds:
                 cred_dict["universe_domain"] = secret_creds["universe_domain"]

            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except KeyError as ke: 
            st.error(f"Secrets Key Error during Firebase init: '{ke}'. Check .streamlit/secrets.toml.")
            return None 
        except Exception as e1: 
            st.error(f"Failed to initialize Firebase from secrets: {e1}. Trying local file...")
            repo_root = os.path.dirname(__file__)
            local_cred_path = os.path.join(repo_root, "firebase-service-account-key.json")
            try:
                if not os.path.exists(local_cred_path):
                    st.error(f"Local Firebase credentials file not found at: {local_cred_path}. Firebase NOT initialized.")
                    return None
                cred = credentials.Certificate(local_cred_path)
                firebase_admin.initialize_app(cred) 
            except Exception as e2: 
                st.error(f"Firebase Local File Error: {e2}. Firebase NOT initialized. Original secrets error was: {e1}")
                return None 
    
    if firebase_admin._apps:
        try:
            return firestore.client()
        except Exception as e:
            st.error(f"Error getting Firestore client after initialization: {e}")
            return None
    return None

def _clean_data_for_firestore(data_dict):
    cleaned_data = {}
    for key, value in data_dict.items():
        if pd.isna(value):
            cleaned_data[key] = None  
        elif isinstance(value, pd.Timestamp):
            cleaned_data[key] = value.to_pydatetime() 
        elif isinstance(value, (int, float, bool, str, list, dict, bytes, datetime, credentials.Certificate)): 
            cleaned_data[key] = value
        elif isinstance(value, (pd.Series, pd.DataFrame)):
             cleaned_data[key] = None 
        else:
            try:
                cleaned_data[key] = str(value) 
            except Exception:
                cleaned_data[key] = None
    return cleaned_data

def add_stock_to_firestore(db, stock_data):
    if not db:
        return False, "Firestore not initialized"
    try:
        ticker_str = str(stock_data.get("Ticker", "")).strip()
        if not ticker_str:
            return False, "Ticker symbol is missing or empty."
        cleaned_stock_data = _clean_data_for_firestore(stock_data)
        doc_ref = db.collection("portfolios").document(USER_ID_PLACEHOLDER).collection("stocks").document(ticker_str)
        doc_ref.set(cleaned_stock_data)
        return True, None
    except Exception as e:
        return False, str(e)

def get_portfolio_from_firestore(db):
    if not db:
        return pd.DataFrame(), "Firestore not initialized"
    try:
        stocks_ref = db.collection("portfolios").document(USER_ID_PLACEHOLDER).collection("stocks")
        docs = stocks_ref.stream()
        portfolio_list = []
        for doc in docs:
            stock_item = doc.to_dict()
            stock_item['Ticker'] = doc.id 
            if 'Purchase_Date' in stock_item and stock_item['Purchase_Date'] is not None:
                if not isinstance(stock_item['Purchase_Date'], datetime):
                    try: 
                        stock_item['Purchase_Date'] = pd.to_datetime(stock_item['Purchase_Date']).to_pydatetime()
                    except ValueError: 
                        stock_item['Purchase_Date'] = None
            portfolio_list.append(stock_item)

        if not portfolio_list:
            return pd.DataFrame(), None 

        df = pd.DataFrame(portfolio_list)
        
        # Assuming DEFAULT_PORTFOLIO_COLS is defined in the calling module (ui_portfolio_page.py)
        # or passed as an argument for better modularity. For now, this will cause an error
        # if this file is run standalone without DEFAULT_PORTFOLIO_COLS being in its scope.
        # A better practice would be to pass DEFAULT_PORTFOLIO_COLS to this function.
        # from ui_portfolio_page import DEFAULT_PORTFOLIO_COLS # This would create circular import if not careful

        # For now, let's assume these cols are known or we just return the df as is from Firestore
        # The calling function (in ui_portfolio_page) can handle ensuring default columns.
        
        numeric_cols = ['Purchase_Price', 'Quantity', 'Investment_Value', 'CMP', 'MarketCap', 'ATH', 
                        'ATH_Correction_Percent', 'Current_Value', 'Profit_Loss', 
                        'Profit_Loss_Percent', 'Day_Change_Stock_Percent', 'Percent_of_Portfolio']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if 'Purchase_Date' in df.columns:
             df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce') 
        
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def delete_stock_from_firestore(db, ticker_symbol):
    if not db:
        return False, "Firestore not initialized"
    try:
        ticker_str = str(ticker_symbol).strip()
        if not ticker_str:
            return False, "Ticker symbol is missing or empty."
        doc_ref = db.collection("portfolios").document(USER_ID_PLACEHOLDER).collection("stocks").document(ticker_str)
        doc_ref.delete()
        return True, None
    except Exception as e:
        return False, str(e)

# --- NEW Functions for Portfolio Value History in Firestore ---
@st.cache_data(ttl=5*60) # Cache for 5 mins, or adjust as needed
def load_portfolio_value_history_from_firestore(_db, user_id=USER_ID_PLACEHOLDER):
    """Loads portfolio value history from Firestore and returns a sorted DataFrame."""
    if not _db:
        return pd.DataFrame(columns=["Date", "TotalValue"]), "Firestore not initialized"
    try:
        history_ref = _db.collection("users").document(user_id).collection(PORTFOLIO_HISTORY_COLLECTION).order_by("date_str")
        docs = history_ref.stream()
        history_data = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure date_str is present and valid before conversion
            if "date_str" in data and data["date_str"]:
                try:
                    history_data.append({
                        "Date": pd.to_datetime(data["date_str"]), 
                        "TotalValue": data.get("total_value") # Use .get for safety
                    })
                except ValueError:
                    # Skip entries with unparseable dates
                    # print(f"Warning: Could not parse date_str '{data['date_str']}' from portfolio history.")
                    pass 
            
        if not history_data:
            return pd.DataFrame(columns=["Date", "TotalValue"]), None # No error if just empty
        
        df = pd.DataFrame(history_data)
        if not df.empty:
            df.sort_values(by="Date", inplace=True)
        return df, None
    except Exception as e:
        return pd.DataFrame(columns=["Date", "TotalValue"]), str(e)

def log_portfolio_value_to_firestore(db, user_id, log_date_str, total_value):
    """Logs or updates the portfolio value for a specific date in Firestore."""
    if not db:
        return False, "Firestore not initialized"
    try:
        # Use the date string as the document ID for easy lookup and overwrite
        doc_ref = db.collection("users").document(user_id).collection(PORTFOLIO_HISTORY_COLLECTION).document(log_date_str)
        doc_ref.set({
            "date_str": log_date_str, 
            "total_value": total_value,
            "logged_at": firestore.SERVER_TIMESTAMP 
        }, merge=True) # Use merge=True to update if exists, or create if not
        return True, None
    except Exception as e:
        return False, str(e)