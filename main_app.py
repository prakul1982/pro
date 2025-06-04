# dontknwo/main_app.py
import streamlit as st
import pandas as pd

# --- Set Page Config ONCE and FIRST ---
st.set_page_config(layout="wide", page_title="Market Analysis Dashboard")

# Import page rendering functions
from ui_home_page import render_home_page
from ui_market_cycles_page import display_market_cycles_page 
from ui_data_management_hub import render_data_management_page
from ui_forecasting_lab import render_forecasting_lab_page
from ui_portfolio_page import render_portfolio_page 

# Attempt to import firebase_utils
try:
    from firebase_utils import initialize_firebase_admin
    firebase_utils_imported = True
    # print("main_app.py: Successfully imported initialize_firebase_admin from firebase_utils.") # Removed
except ImportError:
    initialize_firebase_admin = None
    firebase_utils_imported = False
    # print("WARNING: main_app.py: firebase_utils.py not found or initialize_firebase_admin cannot be imported. Firebase features will be disabled.") # Removed
except Exception as e:
    initialize_firebase_admin = None
    firebase_utils_imported = False
    # print(f"ERROR: main_app.py: Unexpected error importing from firebase_utils: {e}") # Removed


def main():
    db_client = None

    if firebase_utils_imported and initialize_firebase_admin is not None:
        # print("main_app.py: Calling initialize_firebase_admin() to get db_client...") # Removed
        db_client = initialize_firebase_admin()
        
        if db_client:
            pass # Successfully initialized
            # print("main_app.py: db_client obtained successfully.") # Removed
        else:
            # print("main_app.py: initialize_firebase_admin() returned None. Firebase initialization failed.") # Removed
            # The st.error in initialize_firebase_admin should cover UI feedback for init failure.
            # This sidebar error is a good summary if init specifically returned None here.
            st.sidebar.error("🚨 Firebase Connection Failed! Portfolio features may be limited. Check terminal logs.")
    else:
        # print("main_app.py: firebase_utils not imported or initialize_firebase_admin is None. Portfolio won't use DB.") # Removed
        st.sidebar.warning("Firebase utilities not found/import error. Portfolio will not persist.")

    st.sidebar.title("Navigation")
    
    general_session_keys = {
        'new_fii_entries': [],
        'fii_data_editor_df': pd.DataFrame(),
        'corrections_editor_df': pd.DataFrame()
    }
    for key, default_value in general_session_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    app_page_options = ["Home", "Market Cycles", "Data Management Hub", "Forecasting Lab"]
    
    if db_client is not None:
        app_page_options.append("My Portfolio")
    else:
        # This provides a clear indication in the UI that the portfolio page might have issues
        app_page_options.append("My Portfolio (DB Error)") 

    app_page = st.sidebar.radio(
        "Go to",
        app_page_options,
        key="main_nav_radio_v3_corrected" 
    )

    if app_page == "Home":
        render_home_page()
    elif app_page == "Market Cycles":
        display_market_cycles_page() 
    elif app_page == "Data Management Hub":
        render_data_management_page()
    elif app_page == "Forecasting Lab":
        render_forecasting_lab_page()
    elif app_page == "My Portfolio":
        if db_client:
            render_portfolio_page(db_client)
        else:
            # This case should ideally be less frequent if "My Portfolio (DB Error)" is shown
            st.error("Critical Error: Portfolio page selected but DB client is not available.")
            st.info("Please check Firebase setup and terminal logs. The app tried to initialize Firebase, but it failed.")
    elif app_page == "My Portfolio (DB Error)":
        st.error("Portfolio feature is currently unavailable due to a database connection issue or missing Firebase configuration.")
        st.info("Please check terminal logs for specific Firebase initialization errors and ensure your '.streamlit/secrets.toml' (if deploying) or local 'firebase-service-account-key.json' (if running locally and secrets failed) and 'firebase_utils.py' are correctly set up.")

if __name__ == "__main__":
    main()