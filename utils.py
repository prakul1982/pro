# dontknow/utils.py
import streamlit as st
import pandas as pd
from datetime import datetime, date
import re
import os

# --- Constants for CSV paths (Examples, actual paths might be defined in page files or a config module) ---
# DAILY_FII_DII_RAW_CSV_PATH = "daily_fii_dii_scraped_data.csv"
FII_DATA_CSV_PATH = "historical_fii_data.csv"
NIFTY_PE_HISTORY_CSV_PATH = "nifty_pe_ratio_live.csv"

def safe_get(data_dict, key, default='N/A'):
    """Safely get a value from a dictionary."""
    return data_dict.get(key, default) if isinstance(data_dict, dict) else default

def format_large_number(num, default='N/A'):
    """Formats large numbers into Lakhs or Crores, or keeps original formatting for smaller numbers."""
    if num is None or (isinstance(num, str) and num == 'N/A') or not isinstance(num, (int, float)):
        return default
    if abs(num) >= 1_00_00_000: # Crores
        return f"₹{num/1_00_00_000:.2f} Cr"
    elif abs(num) >= 1_00_000: # Lakhs
         return f"₹{num/1_00_000:.2f} L"
    return f"₹{num:,.2f}" if isinstance(num, float) else f"₹{num:,}"

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

def parse_date_from_string(date_str):
    """
    Attempts to parse date from various string formats.
    Returns a datetime.date object or None.
    """
    if pd.isna(date_str): return None
    if isinstance(date_str, (datetime, date, pd.Timestamp)):
        return pd.to_datetime(date_str).date()
    
    if not isinstance(date_str, str): return None

    try:
        dt_obj = pd.to_datetime(date_str, errors='raise')
        return dt_obj.date()
    except (ValueError, TypeError):
        pass 

    match_obj_repr = re.match(r"datetime.date\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)", date_str)
    if match_obj_repr:
        try:
            year, month, day_val = map(int, match_obj_repr.groups())
            return date(year, month, day_val)
        except ValueError:
            pass 

    try:
        return datetime.strptime(date_str, '%d-%b-%Y').date()
    except ValueError:
        pass
    
    # print(f"DEBUG: utils.parse_date_from_string: Could not parse date string: '{date_str}' with known formats.") # Removed
    return None

def calculate_rsi(data, window=14):
    if data is None or not isinstance(data, pd.Series) or len(data) < window + 1:
        return pd.Series(dtype='float64', index=data.index if isinstance(data, pd.Series) else None)
    
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=window - 1, adjust=False, min_periods=window -1).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=window - 1, adjust=False, min_periods=window -1).mean()
    
    rs = gain / loss
    rs = rs.replace([float('inf'), -float('inf')], float('nan'))
    
    rsi = 100 - (100 / (1 + rs))
    rsi.loc[loss == 0] = 100 
    rsi.loc[(gain == 0) & (loss == 0)] = 50 
    
    return rsi.fillna(50)

def calculate_macd(data, window_slow=26, window_fast=12, window_signal=9):
    if data is None or not isinstance(data, pd.Series) or len(data) < window_slow:
        empty_series = pd.Series(dtype='float64', index=data.index if isinstance(data, pd.Series) else None)
        return empty_series, empty_series, empty_series
        
    exp1 = data.ewm(span=window_fast, adjust=False).mean()
    exp2 = data.ewm(span=window_slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=window_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    if data is None or not isinstance(data, pd.Series) or len(data) < window:
        empty_series = pd.Series(dtype='float64', index=data.index if isinstance(data, pd.Series) else None)
        return empty_series, empty_series, empty_series
        
    sma = data.rolling(window=window, min_periods=window).mean()
    std_dev = data.rolling(window=window, min_periods=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, sma, lower_band

def load_csv_data(file_path, default_cols, parse_dates_cols=None, error_level="warning", thousands=None):
    """
    Loads data from a CSV file. If file not found or empty, returns an empty DataFrame
    with default_cols and appropriate dtypes.
    Adds missing default_cols to the loaded DataFrame if it's not empty.
    """
    numeric_keywords = ['investment', 'value', 'drop', 'net', 'price', 'qty', 'fii_total_net_investment', 
                        'pe_ratio', 'cmp', 'ath', 'fii_net_purchase_sales', 'dii_net_purchase_sales',
                        'profit_loss', 'marketcap', 'day_change', 'percent_of_portfolio', 'quantity']
    
    try:
        df = pd.read_csv(file_path, thousands=thousands)
        
        for col in default_cols:
            if col not in df.columns:
                if col in (parse_dates_cols or []):
                    df[col] = pd.NaT 
                elif any(kw in col.lower() for kw in numeric_keywords):
                    df[col] = 0.0 
                else:
                    df[col] = "" 

        if parse_dates_cols:
            for col in parse_dates_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
        
        for col in default_cols:
            if any(kw in col.lower() for kw in numeric_keywords) and col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0.0)
        return df

    except FileNotFoundError:
        # print(f"INFO: File `{file_path}` not found. Creating an empty DataFrame with default schema.") # Removed
        pass
    except pd.errors.EmptyDataError:
        # print(f"INFO: File `{file_path}` is empty. Creating an empty DataFrame with default schema.") # Removed
        pass
    except Exception as e:
        st.error(f"Critical error loading data from `{file_path}`: {e}") # Keep st.error for critical load failure
    
    df = pd.DataFrame()
    for col in default_cols:
        if col in (parse_dates_cols or []):
            df[col] = pd.Series(dtype='datetime64[ns]')
        elif any(kw in col.lower() for kw in numeric_keywords):
            df[col] = pd.Series(dtype='float64')
        else:
            df[col] = pd.Series(dtype='object')
    return df

def save_csv_data(df, file_path, date_cols_to_format=None):
    try:
        df_to_save = df.copy()
        if date_cols_to_format:
            for col in date_cols_to_format:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_datetime(df_to_save[col], errors='coerce')
                    df_to_save[col] = df_to_save[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '')
        
        for col in df_to_save.columns:
            if pd.api.types.is_numeric_dtype(df_to_save[col]):
                df_to_save[col] = df_to_save[col].fillna(0.0)
            elif pd.api.types.is_object_dtype(df_to_save[col]): 
                if not (date_cols_to_format and col in date_cols_to_format):
                    df_to_save[col] = df_to_save[col].fillna('')

        df_to_save.to_csv(file_path, index=False)
        # print(f"Data successfully saved to `{os.path.basename(file_path)}`") # Removed
        return True
    except Exception as e:
        # print(f"Error saving data to `{file_path}`: {e}") # Removed
        # If an error occurs, it's better to let the calling function handle UI messages (e.g., st.error)
        # and this utility function can just signal failure.
        return False

def append_to_daily_fii_dii_csv(new_daily_data_df, file_path):
    """
    Appends new daily FII/DII data to the existing CSV file, ensuring 'Date' uniqueness.
    Enforces the output CSV to have columns: 'Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'.
    """
    if new_daily_data_df is None or new_daily_data_df.empty:
        # print("DEBUG: append_to_daily_fii_dii_csv: No new data to append.") # Removed
        return False
    
    expected_cols = ['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales']
    
    # Ensure the input DataFrame at least has the columns, even if some are all NA
    # The function get_cached_latest_fpi_data in market_cycles_page.py should already ensure this structure.
    for col in expected_cols:
        if col not in new_daily_data_df.columns:
            # This case should ideally be handled by the data source or get_cached_latest_fpi_data
            # print(f"ERROR: append_to_daily_fii_dii_csv: Input DataFrame missing critical column: {col}") # Removed
            return False # Cannot proceed without expected columns

    try:
        # Prepare new data: select only expected columns, convert types
        new_df_filtered = new_daily_data_df[expected_cols].copy()
        new_df_filtered['Date'] = pd.to_datetime(new_df_filtered['Date'], errors='coerce')
        new_df_filtered.dropna(subset=['Date'], inplace=True)
        if new_df_filtered.empty:
             # print("DEBUG: append_to_daily_fii_dii_csv: New data is empty after date processing.") # Removed
            return False # No valid new data to append

        for col in ['FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales']:
            if col in new_df_filtered.columns: # Should always be true due to earlier check
                new_df_filtered[col] = pd.to_numeric(new_df_filtered[col], errors='coerce')
        
        combined_df = new_df_filtered # Initialize with new, filtered data

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                existing_df = pd.read_csv(file_path) 
                if 'Date' in existing_df.columns:
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
                
                # Ensure existing_df also has the expected columns for clean concatenation
                # Fill missing expected columns with NA before filtering
                for col in expected_cols:
                    if col not in existing_df.columns:
                        existing_df[col] = pd.NA
                existing_df_filtered = existing_df[expected_cols].copy() # Select and order

                combined_df = pd.concat([existing_df_filtered, new_df_filtered], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass # combined_df remains new_df_filtered
            except Exception as read_e:
                # print(f"WARNING: Error reading existing FII/DII CSV {file_path}: {read_e}. Overwriting with new data.") # Removed
                pass # combined_df remains new_df_filtered (effectively overwriting on error)
        
        combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        combined_df.sort_values(by='Date', inplace=True)
        
        # Ensure the final DataFrame written to CSV has only the expected columns
        final_output_df = combined_df[expected_cols].copy()
        final_output_df['Date'] = final_output_df['Date'].dt.strftime('%Y-%m-%d') # Format date for saving
        
        final_output_df.to_csv(file_path, index=False)
        # print(f"DEBUG: Successfully appended/updated data in {file_path}") # Removed
        return True
    except Exception as e:
        # print(f"ERROR: append_to_daily_fii_dii_csv for {file_path}: {e}") # Removed
        # The calling function in the UI page should use st.error()
        return False