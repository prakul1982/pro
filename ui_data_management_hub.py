import streamlit as st
import pandas as pd
import os
from datetime import datetime
from data_fetcher import fetch_nifty_pe_historical_data, fetch_fii_dii_data_nse # Ensure these are correct

# --- Constants ---
PE_FILE_PATH = 'nifty_pe_ratio_live.csv'
FII_DII_DAILY_CSV_PATH = 'daily_fii_dii_nse_data.csv'
MARKET_CORRECTIONS_CSV_PATH = "market_corrections_data.csv"

# Expected columns for creating empty DataFrames if files don't exist or are empty
# MODIFIED FII_DII_DEFAULT_COLS
FII_DII_DEFAULT_COLS = ['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'] 
PE_DEFAULT_COLS = ['Date', 'PE_Ratio']
CORRECTIONS_DEFAULT_COLS = ["Event_Name", "Approx_Peak_Month_Str", "Approx_Trough_Month_Str",
                            "Leading_Indicators_Signals", "FII_Behavior_Qualitative", "Key_Reasons"]

# --- Helper Functions ---

def get_file_update_time(file_path):
    if os.path.exists(file_path):
        return datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    return "File not found"

def load_csv_data_for_editing(file_path, default_cols, parse_dates_cols=None, thousands=None):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, thousands=thousands)
            if df.empty:
                return pd.DataFrame(columns=default_cols)
            if parse_dates_cols:
                for col in parse_dates_cols:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except Exception:
                             pass 
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=default_cols)
        except Exception as e:
            st.error(f"Error loading {file_path} for editing: {e}")
            return pd.DataFrame(columns=default_cols)
    return pd.DataFrame(columns=default_cols)

def save_df_to_csv(df, file_path, desired_columns=None):
    """Saves DataFrame to CSV, optionally filtering to desired_columns."""
    try:
        if desired_columns:
            # Ensure only desired columns that actually exist in df are saved, in order
            cols_to_save = [col for col in desired_columns if col in df.columns]
            df_to_save = df[cols_to_save]
        else:
            df_to_save = df
        df_to_save.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to {file_path}: {e}")
        return False

# REFINED append_to_daily_fii_dii_csv
def append_to_daily_fii_dii_csv(new_data_df, file_path=FII_DII_DAILY_CSV_PATH):
    if new_data_df is None or new_data_df.empty:
        return False # Nothing to append

    # Define the exact columns we want in the final CSV
    desired_columns = ['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales']

    try:
        # Validate and prepare new data
        if 'Date' not in new_data_df.columns:
            st.error("New FII/DII data from fetch is missing 'Date' column.")
            return False
        # Ensure the new data has all desired columns, fill with NaN if not (though fetch should provide them)
        for col in desired_columns:
            if col not in new_data_df.columns:
                new_data_df[col] = pd.NA 
        
        new_data_df_filtered = new_data_df[desired_columns].copy() # Select and order
        new_data_df_filtered['Date'] = pd.to_datetime(new_data_df_filtered['Date'], errors='coerce')
        new_data_df_filtered.dropna(subset=['Date'], inplace=True)

        if new_data_df_filtered.empty:
            st.info("No valid new FII/DII data rows to append after date processing.")
            return False

        combined_df = new_data_df_filtered # Initialize with new data

        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                if not existing_df.empty and 'Date' in existing_df.columns:
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
                    
                    # Ensure existing_df also only has desired columns for concat
                    # If some are missing, they will be NaN and handled by concat
                    existing_cols_present = [col for col in desired_columns if col in existing_df.columns]
                    existing_df_filtered = existing_df[existing_cols_present].copy()
                    
                    combined_df = pd.concat([existing_df_filtered, new_data_df_filtered])
                # If existing_df is empty or has no Date, combined_df remains new_data_df_filtered
            except pd.errors.EmptyDataError: 
                pass # combined_df is already new_data_df_filtered
            except Exception as e_read: 
                st.warning(f"Could not properly read existing FII/DII CSV {file_path}: {e_read}. It might be overwritten if new data is valid.")
        
        # Drop duplicates based on 'Date', keeping the last entry (preferring new data)
        combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        
        # Final filter and ordering for the output CSV
        combined_df = combined_df[desired_columns] 
        
        combined_df.sort_values(by='Date', inplace=True)
        combined_df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error in append_to_daily_fii_dii_csv for {file_path}: {e}")
        return False


def render_data_management_page():
    st.title("🛠️ Data Management Hub")
    st.markdown("Manage the underlying data files for the dashboards. You can refresh data from sources, upload new CSV files, or edit existing data directly.")
    st.markdown("---")

    # --- NIFTY P/E Ratio Data Management ---
    st.subheader("NIFTY 50 P/E Ratio Data")
    st.markdown(f"**Source:** Finlive.in | **File:** `{PE_FILE_PATH}` | **Last Updated:** {get_file_update_time(PE_FILE_PATH)}")
    st.caption("Expected CSV format: Columns 'Date' (YYYY-MM-DD or parsable date format) and 'PE_Ratio' (numeric). Edit directly below or upload a new file.")

    pe_df = load_csv_data_for_editing(PE_FILE_PATH, default_cols=PE_DEFAULT_COLS, parse_dates_cols=['Date'])
    
    col1_pe_actions, col2_pe_actions = st.columns(2)
    with col1_pe_actions:
        if st.button("🔄 Refresh P/E Data from Finlive.in", key="refresh_pe_hub_finlive_dm_edit", help="Fetches the latest data from the source and overwrites the local file."):
            with st.spinner("Fetching NIFTY P/E data from Finlive.in..."):
                df_new_pe = fetch_nifty_pe_historical_data()
                if df_new_pe is not None and not df_new_pe.empty:
                    if save_df_to_csv(df_new_pe, PE_FILE_PATH): # save_df_to_csv saves all columns from df_new_pe
                        st.success(f"NIFTY P/E data refreshed and saved to {PE_FILE_PATH}")
                        st.rerun()
                else:
                    st.error("Failed to fetch NIFTY P/E data. Check console for scraper logs if Selenium is used, or the source might be unavailable.")
    
    with col2_pe_actions:
        uploaded_pe_file = st.file_uploader(f"Upload to replace `{PE_FILE_PATH}`", type=['csv'], key="upload_pe_hub_dm_edit")
        if uploaded_pe_file is not None:
            try:
                df_upload_pe = pd.read_csv(uploaded_pe_file)
                if 'Date' in df_upload_pe.columns and 'PE_Ratio' in df_upload_pe.columns:
                    # For PE data, we assume the uploaded file is already in the desired simple format
                    save_df_to_csv(df_upload_pe, PE_FILE_PATH)
                    st.success(f"Uploaded P/E data saved to {PE_FILE_PATH}")
                    st.rerun()
                else:
                    st.error("Uploaded P/E CSV must contain 'Date' and 'PE_Ratio' columns.")
            except Exception as e:
                st.error(f"Error processing uploaded P/E file: {e}")

    st.markdown("##### Edit NIFTY P/E Data:")
    edited_pe_df = st.data_editor(pe_df, key="pe_editor", num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save P/E Changes", key="save_pe_hub"):
        if save_df_to_csv(edited_pe_df, PE_FILE_PATH): # Saves all columns from the editor
            st.success(f"Changes saved to {PE_FILE_PATH}")
            st.rerun() 

    if os.path.exists(PE_FILE_PATH):
        with open(PE_FILE_PATH, "rb") as fp:
            st.download_button(
                label=f"📥 Download {PE_FILE_PATH}",
                data=fp,
                file_name=os.path.basename(PE_FILE_PATH),
                mime="text/csv",
                key="download_pe_hub_dm_edit"
            )
    st.markdown("---")


    # --- Daily FII/DII Data Management (NSE) ---
    st.subheader("Daily FII/DII Data (NSE)")
    st.markdown(f"**Source:** NSE (Experimental Scraper) | **File:** `{FII_DII_DAILY_CSV_PATH}` | **Last Updated:** {get_file_update_time(FII_DII_DAILY_CSV_PATH)}")
    # Caption now reflects the 3-column structure for fetched data
    st.caption(f"Expected CSV format for fetched data: Columns 'Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'. Editor allows manual changes.")

    # Uses the MODIFIED FII_DII_DEFAULT_COLS for consistency when file is absent
    fii_dii_df = load_csv_data_for_editing(FII_DII_DAILY_CSV_PATH, default_cols=FII_DII_DEFAULT_COLS, parse_dates_cols=['Date'])

    col1_fii_actions, col2_fii_actions = st.columns(2)
    with col1_fii_actions:
        if st.button("🔄 Refresh Daily FII/DII Data (from NSE)", key="refresh_fii_dii_hub_dm_edit", help="Fetches the latest data from NSE and saves only Date, FII_Net_Purchase_Sales, DII_Net_Purchase_Sales."):
            with st.spinner("Fetching Daily FII/DII data from NSE..."):
                df_new_fii = fetch_fii_dii_data_nse() 
                if df_new_fii is not None and not df_new_fii.empty:
                    if append_to_daily_fii_dii_csv(df_new_fii.copy(), FII_DII_DAILY_CSV_PATH):
                        st.success(f"Daily FII/DII data refreshed. {FII_DII_DAILY_CSV_PATH} now contains only Date, FII_Net_Purchase_Sales, DII_Net_Purchase_Sales.")
                        st.rerun()
                    else:
                        st.error(f"Failed to save/append FII/DII data to {FII_DII_DAILY_CSV_PATH}. Check previous messages or console.")
                elif df_new_fii is not None and df_new_fii.empty: 
                     st.warning("No new FII/DII data was fetched from NSE. The source might not have new data or the scraper needs an update.")
                else: 
                    st.error("Failed to fetch FII/DII data from NSE. Check console for Selenium logs or ensure Selenium is set up correctly.")
    
    with col2_fii_actions:
        uploaded_fii_file = st.file_uploader(f"Upload to replace `{FII_DII_DAILY_CSV_PATH}`", type=['csv'], key="upload_fii_hub_dm_edit")
        if uploaded_fii_file is not None:
            try:
                df_upload_fii = pd.read_csv(uploaded_fii_file)
                # When uploading, we save what the user provides. The next refresh will re-format to 3 cols.
                # Or, we can enforce 3 columns on upload too. For now, let's allow flexible upload but refresh cleans it.
                if 'Date' in df_upload_fii.columns: 
                    save_df_to_csv(df_upload_fii, FII_DII_DAILY_CSV_PATH) # Saves all columns from upload
                    st.success(f"Uploaded FII/DII data saved to {FII_DII_DAILY_CSV_PATH}. Refresh will format to standard columns.")
                    st.rerun()
                else:
                    st.error("Uploaded CSV for FII/DII data must contain a 'Date' column.")
            except Exception as e:
                st.error(f"Error processing uploaded FII/DII file: {e}")

    st.markdown("##### Edit Daily FII/DII Data:")
    # The data editor will show columns present in the loaded FII_DII_DAILY_CSV_PATH,
    # or FII_DII_DEFAULT_COLS if the file is new/empty.
    edited_fii_dii_df = st.data_editor(fii_dii_df, key="fii_dii_editor", num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save FII/DII Changes", key="save_fii_dii_hub"):
        # Save changes from editor. If user added columns, they will be saved.
        # The next refresh will re-format to the 3 standard columns.
        if save_df_to_csv(edited_fii_dii_df, FII_DII_DAILY_CSV_PATH):
            st.success(f"Changes saved to {FII_DII_DAILY_CSV_PATH}. Refresh will format to standard columns if needed.")
            st.rerun()

    if os.path.exists(FII_DII_DAILY_CSV_PATH):
        with open(FII_DII_DAILY_CSV_PATH, "rb") as fp:
            st.download_button(
                label=f"📥 Download {FII_DII_DAILY_CSV_PATH}",
                data=fp,
                file_name=os.path.basename(FII_DII_DAILY_CSV_PATH),
                mime="text/csv",
                key="download_fii_hub_dm_edit"
            )
    st.markdown("---")

    # --- Market Corrections Data Management ---
    st.subheader("Market Corrections Data")
    st.markdown(f"**File:** `{MARKET_CORRECTIONS_CSV_PATH}` | **Last Updated:** {get_file_update_time(MARKET_CORRECTIONS_CSV_PATH)}")
    st.caption(f"Expected CSV format: Columns like {', '.join(CORRECTIONS_DEFAULT_COLS[:3])}...")

    corrections_df = load_csv_data_for_editing(MARKET_CORRECTIONS_CSV_PATH, default_cols=CORRECTIONS_DEFAULT_COLS)
    
    uploaded_corrections_file = st.file_uploader(f"Upload to replace `{MARKET_CORRECTIONS_CSV_PATH}`", type=['csv'], key="upload_corrections_hub_dm_edit", help="Upload a CSV with market correction details.")
    if uploaded_corrections_file is not None:
        try:
            df_upload_corr = pd.read_csv(uploaded_corrections_file)
            if 'Event_Name' in df_upload_corr.columns: 
                save_df_to_csv(df_upload_corr, MARKET_CORRECTIONS_CSV_PATH)
                st.success(f"Uploaded corrections data saved to {MARKET_CORRECTIONS_CSV_PATH}")
                st.rerun()
            else:
                st.error("Uploaded CSV for Market Corrections data must contain an 'Event_Name' column (or other key identifying columns).")
        except Exception as e:
            st.error(f"Error processing uploaded corrections file: {e}")

    st.markdown("##### Edit Market Corrections Data:")
    edited_corrections_df = st.data_editor(corrections_df, key="corrections_editor", num_rows="dynamic", use_container_width=True)
    if st.button("💾 Save Corrections Changes", key="save_corrections_hub"):
        if save_df_to_csv(edited_corrections_df, MARKET_CORRECTIONS_CSV_PATH):
            st.success(f"Changes saved to {MARKET_CORRECTIONS_CSV_PATH}")
            st.rerun()

    if os.path.exists(MARKET_CORRECTIONS_CSV_PATH):
        with open(MARKET_CORRECTIONS_CSV_PATH, "rb") as fp:
            st.download_button(
                label=f"📥 Download {MARKET_CORRECTIONS_CSV_PATH}",
                data=fp,
                file_name=os.path.basename(MARKET_CORRECTIONS_CSV_PATH),
                mime="text/csv",
                key="download_corrections_hub_dm_edit"
            )