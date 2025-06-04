# daily_portfolio_logger.py
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# --- IMPORT YOUR ACTUAL UTILITY FUNCTIONS ---
# Adjust paths if your logger script is in a different directory than these utils
try:
    from firebase_utils import (
        get_portfolio_from_firestore, 
        log_portfolio_value_to_firestore, 
        USER_ID_PLACEHOLDER,
        PORTFOLIO_HISTORY_COLLECTION # If you defined it there
    )
    # If initialize_logger_firebase is also in firebase_utils, import it too,
    # otherwise keep the local version defined below.
except ImportError:
    print("ERROR: Could not import from firebase_utils. Ensure it's in the PYTHONPATH or same directory.")
    # Define USER_ID_PLACEHOLDER and PORTFOLIO_HISTORY_COLLECTION here if not imported
    USER_ID_PLACEHOLDER = "default_user_portfolio" 
    PORTFOLIO_HISTORY_COLLECTION = "portfolio_value_history"
    # You would need to copy the actual functions here if import fails, which is not ideal

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_KEY_PATH = SERVICE_ACCOUNT_KEY_PATH = "firebase-service-account-key.json"
# USER_ID is now imported or defined above

# --- Firebase Initialization for this script ---
def initialize_logger_firebase(key_path):
    # Give a unique name to this Firebase app instance
    app_name = f"portfolioLoggerApp-{os.getpid()}"
    if not firebase_admin._apps or app_name not in firebase_admin._apps: # Check if this specific app is initialized
        try:
            if not os.path.exists(key_path):
                print(f"Logger ERROR: Service account key not found at resolved path: {key_path}")
                return None
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, name=app_name)
            print(f"Logger: Firebase Admin SDK initialized (app: {app_name}).")
        except Exception as e:
            print(f"Logger ERROR: Firebase Admin SDK initialization failed for app {app_name}: {e}")
            return None
    return firestore.client(firebase_admin.get_app(name=app_name))

# --- Helper functions (keep specific logger helpers or import from utils) ---
def get_current_price_for_logger(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="2d") # Fetch 2 days to be safe for recent close
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        # Fallback if history is empty
        info = stock.info
        cmp = info.get('currentPrice', info.get('regularMarketPreviousClose', info.get('previousClose')))
        return float(cmp) if cmp is not None else 0.0
    except Exception as e:
        print(f"Logger WARN: Could not fetch price for {ticker_symbol}: {e}")
        return 0.0

# --- Main Logging Logic ---
def run_daily_log():
    db = initialize_logger_firebase(SERVICE_ACCOUNT_KEY_PATH)
    if not db:
        print("Logger: Exiting, DB not initialized for logger.")
        return

    print(f"Logger: Script started at {datetime.now()}")
    print(f"Logger: Fetching portfolio for user: {USER_ID_PLACEHOLDER}")

    try:
        # Use the imported get_portfolio_from_firestore
        # Ensure it's the version that takes (db) or (db, user_id)
        # Assuming your firebase_utils.get_portfolio_from_firestore takes (db, user_id)
        # If it only takes (db), adjust call or function.
        # The version I provided for firebase_utils.py takes (db) and uses global USER_ID_PLACEHOLDER
        portfolio_df, error = get_portfolio_from_firestore(db) # Assuming it uses the USER_ID_PLACEHOLDER internally

        if error:
            print(f"Logger: Error fetching portfolio for user {USER_ID_PLACEHOLDER}: {error}")
            return
        if portfolio_df.empty:
            print(f"Logger: Portfolio for user {USER_ID_PLACEHOLDER} is empty.")
            return

        total_current_value = 0.0
        print(f"Logger: Calculating current values for {len(portfolio_df)} stocks...")
        for index, row in portfolio_df.iterrows():
            ticker = row.get('Ticker') 
            quantity = float(row.get('Quantity', 0.0))
            if ticker and quantity > 0:
                cmp = get_current_price_for_logger(ticker)
                if cmp > 0: 
                    total_current_value += cmp * quantity
                # else: # Already printed by get_current_price_for_logger
                    # print(f"Logger WARN: CMP for {ticker} is 0 or invalid, excluded from total value.")
            # else: # Already printed by get_current_price_for_logger
                # print(f"Logger WARN: Skipping stock with missing Ticker or zero Quantity: {row}")

        print(f"Logger: Total current portfolio value calculated: {total_current_value}")

        if total_current_value > 0:
            today_str = datetime.now().strftime("%Y-%m-%d")

            # Use the imported log_portfolio_value_to_firestore
            success, msg = log_portfolio_value_to_firestore(db, USER_ID_PLACEHOLDER, today_str, total_current_value)
            if success:
                print(f"Logger: Successfully logged portfolio value for {today_str} to Firestore: {total_current_value}")
            else:
                print(f"Logger: Failed to log portfolio value: {msg}")
        else:
            print("Logger: Total portfolio value is zero or less, not logging.")

    except Exception as e:
        print(f"Logger ERROR: An error occurred during daily log: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_daily_log()
    print(f"Logger: Finished daily portfolio value log at {datetime.now()}")