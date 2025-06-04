# ui_portfolio_page.py
import streamlit as st
import pandas as pd
from datetime import date, datetime, timezone 
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go 
import os

# Assuming these imports are correctly set up in your project structure
from data_fetcher import get_stock_history
from utils import safe_get, format_large_number
from firebase_utils import (
    get_portfolio_from_firestore,
    add_stock_to_firestore,
    delete_stock_from_firestore,
    log_portfolio_value_to_firestore,
    load_portfolio_value_history_from_firestore,
    USER_ID_PLACEHOLDER
)

# --- Constants ---
DEFAULT_PORTFOLIO_COLS = [
    "Ticker", "Company_Name", "Purchase_Date", "Purchase_Price", "Quantity",
    "Notes", "Stop_Loss", "Target_Price", "Investment_Value",
    "CMP", "MarketCap", "Category", "Sector",
    "ATH", "ATH_Correction_Percent", "Signal",
    "Current_Value", "Profit_Loss", "Profit_Loss_Percent",
    "Day_Change_Stock_Percent", "Percent_of_Portfolio"
]

LARGE_CAP_THRESHOLD = 20000 * 10**7
MID_CAP_THRESHOLD = 5000 * 10**7
ATH_BUY_TRIGGERS = {
    "Large-Cap": -20.0, "Mid-Cap": -35.0, "Small-Cap": -55.0, "Unknown": -100.0
}
PROFIT_SELL_TRIGGERS = {
    "Large-Cap": 50.0, "Mid-Cap": 75.0, "Small-Cap": 100.0, "Unknown": 200.0
}
# PORTFOLIO_VALUE_HISTORY_CSV constant removed, using Firestore

# --- Helper Functions ---
def safe_to_float(value, default=None):
    if value is None: return default
    try:
        if isinstance(value, str):
            cleaned_value = value.replace(',', '')
            if any(c.isalpha() for c in cleaned_value if c.lower() not in ['e']): return default
            return float(cleaned_value)
        return float(value)
    except (ValueError, TypeError): return default

@st.cache_data(ttl=300)
def fetch_stock_data_for_portfolio(ticker_symbol):
    clean_ticker_symbol = str(ticker_symbol).strip()
    stock_obj = yf.Ticker(clean_ticker_symbol)
    info = {}
    try:
        info = stock_obj.info
        if not info or not info.get('symbol'):
            hist_fallback = stock_obj.history(period="1d")
            if not hist_fallback.empty and 'Close' in hist_fallback:
                cmp_fallback = safe_to_float(hist_fallback['Close'].iloc[-1])
                return cmp_fallback, None, None, None, None, "Unknown"
            return None, None, None, None, None, None
    except Exception:
        try:
            hist_fallback = stock_obj.history(period="1d")
            if not hist_fallback.empty and 'Close' in hist_fallback:
                cmp_fallback = safe_to_float(hist_fallback['Close'].iloc[-1])
                return cmp_fallback, None, None, None, None, "Unknown"
        except Exception:
            pass
        return None, None, None, None, None, None

    cmp_raw = safe_get(info, 'currentPrice', safe_get(info, 'regularMarketPrice', safe_get(info, 'regularMarketPreviousClose', safe_get(info, 'previousClose'))))
    previous_close_raw = safe_get(info, 'regularMarketPreviousClose', safe_get(info, 'previousClose'))
    market_open_raw = safe_get(info, 'regularMarketOpen')
    market_cap_raw = safe_get(info, 'marketCap')
    sector_raw = safe_get(info, 'sector', 'N/A')

    cmp = safe_to_float(cmp_raw)
    previous_close = safe_to_float(previous_close_raw)
    market_cap = safe_to_float(market_cap_raw)
    sector = str(sector_raw) if sector_raw not in ['N/A', None] else "Unknown"

    day_change_percent = None
    if cmp is not None and previous_close is not None and previous_close != 0:
        day_change_percent = ((cmp - previous_close) / previous_close) * 100
    elif cmp is not None:
        market_open = safe_to_float(market_open_raw)
        if market_open is not None and market_open != 0:
            day_change_percent = ((cmp - market_open) / market_open) * 100

    ath = None
    try:
        hist_stock_obj_ath = yf.Ticker(clean_ticker_symbol)
        hist_5y = hist_stock_obj_ath.history(period="5y")
        if not hist_5y.empty and 'High' in hist_5y.columns:
            ath = safe_to_float(hist_5y['High'].max())
        else:
            hist_max = hist_stock_obj_ath.history(period="max") 
            if not hist_max.empty and 'High' in hist_max.columns:
                ath = safe_to_float(hist_max['High'].max())
    except Exception:
        ath = None 
    return cmp, ath, day_change_percent, previous_close, market_cap, sector

def get_market_cap_category(market_cap_value):
    if market_cap_value is None: return "Unknown"
    if market_cap_value > LARGE_CAP_THRESHOLD: return "Large-Cap"
    elif market_cap_value > MID_CAP_THRESHOLD: return "Mid-Cap"
    else: return "Small-Cap"

def calculate_portfolio_metrics(portfolio_df_input):
    if portfolio_df_input is None or portfolio_df_input.empty:
        return pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLS), {
            "total_invested": 0, "total_current_value": 0, "overall_pl": 0,
            "overall_pl_percent": 0, "day_portfolio_change_value": 0, "day_portfolio_change_percent": 0
        }
    updated_df = portfolio_df_input.copy()
    for col in DEFAULT_PORTFOLIO_COLS:
        if col not in updated_df.columns:
            if col == 'Purchase_Date': 
                updated_df[col] = pd.NaT
            elif col in ['Purchase_Price', 'Quantity', 'Investment_Value', 'CMP', 'MarketCap', 'ATH', 'ATH_Correction_Percent', 'Current_Value', 'Profit_Loss', 'Profit_Loss_Percent', 'Day_Change_Stock_Percent', 'Percent_of_Portfolio']:
                updated_df[col] = 0.0
            elif col == 'Category' or col == 'Sector': updated_df[col] = "Unknown"
            elif col == 'Signal': updated_df[col] = ""
            else: updated_df[col] = None

    updated_df['Purchase_Price'] = pd.to_numeric(updated_df['Purchase_Price'], errors='coerce').fillna(0.0)
    updated_df['Quantity'] = pd.to_numeric(updated_df['Quantity'], errors='coerce').fillna(0.0)
    updated_df['Investment_Value'] = updated_df['Quantity'] * updated_df['Purchase_Price']
    if 'Purchase_Date' in updated_df.columns: # Ensure Purchase_Date is datetime after potential default column creation
        updated_df['Purchase_Date'] = pd.to_datetime(updated_df['Purchase_Date'], errors='coerce')

    total_investment_value_portfolio = updated_df['Investment_Value'].sum()
    total_current_value_portfolio = 0.0
    total_day_change_value_portfolio = 0.0

    for index, row in updated_df.iterrows():
        ticker_from_row = str(row.get('Ticker','N/A_TICKER')).strip()
        quantity = row.get('Quantity', 0.0)
        current_investment_value = row.get('Investment_Value', 0.0)
        profit_loss_percent_calculated = 0.0
        cmp, ath, day_change_stock_percent, previous_close, market_cap, sector = None, None, None, None, None, "Unknown"

        if not ticker_from_row or pd.isna(ticker_from_row) or ticker_from_row == 'N/A_TICKER':
            for metric_col in ['CMP', 'MarketCap', 'Current_Value', 'ATH', 'ATH_Correction_Percent', 'Day_Change_Stock_Percent']: updated_df.loc[index, metric_col] = 0.0
            updated_df.loc[index, 'Category'] = "Unknown"; updated_df.loc[index, 'Sector'] = "Unknown"; updated_df.loc[index, 'Signal'] = "⚠️ Invalid Ticker"
            updated_df.loc[index, 'Profit_Loss'] = -current_investment_value
            updated_df.loc[index, 'Profit_Loss_Percent'] = -100.0 if current_investment_value != 0 else 0.0
            continue
        try:
            cmp, ath, day_change_stock_percent, previous_close, market_cap, sector = fetch_stock_data_for_portfolio(ticker_from_row)
        except Exception as e:
            st.warning(f"Critical data fetch error for {ticker_from_row} during calculation: {e}.", icon="⚠️")
            for metric_col in ['CMP', 'MarketCap', 'Current_Value', 'ATH', 'ATH_Correction_Percent', 'Day_Change_Stock_Percent']: updated_df.loc[index, metric_col] = 0.0
            updated_df.loc[index, 'Category'] = "Unknown"; updated_df.loc[index, 'Sector'] = "Unknown"; updated_df.loc[index, 'Signal'] = "⚠️ Fetch Error"
            updated_df.loc[index, 'Profit_Loss'] = -current_investment_value
            updated_df.loc[index, 'Profit_Loss_Percent'] = -100.0 if current_investment_value != 0 else 0.0
            continue

        updated_df.loc[index, 'MarketCap'] = market_cap if market_cap is not None else 0.0
        category = get_market_cap_category(market_cap)
        updated_df.loc[index, 'Category'] = category
        updated_df.loc[index, 'Sector'] = sector if (sector and sector != 'N/A') else "Unknown"
        current_signal = "⏳ Hold"

        if cmp is not None:
            updated_df.loc[index, 'CMP'] = cmp
            current_stock_value = quantity * cmp
            updated_df.loc[index, 'Current_Value'] = current_stock_value
            updated_df.loc[index, 'Profit_Loss'] = current_stock_value - current_investment_value
            if current_investment_value != 0:
                profit_loss_percent_calculated = (updated_df.loc[index, 'Profit_Loss'] / current_investment_value) * 100
                updated_df.loc[index, 'Profit_Loss_Percent'] = profit_loss_percent_calculated
            else:
                updated_df.loc[index, 'Profit_Loss_Percent'] = float('inf') if current_stock_value > 0 else (float('-inf') if current_stock_value < 0 else 0.0)
            if day_change_stock_percent is not None:
                updated_df.loc[index, 'Day_Change_Stock_Percent'] = day_change_stock_percent
                if previous_close is not None and previous_close != 0:
                    total_day_change_value_portfolio += (cmp - previous_close) * quantity
            else:
                updated_df.loc[index, 'Day_Change_Stock_Percent'] = 0.0
        else:
            updated_df.loc[index, 'CMP'] = 0.0
            updated_df.loc[index, 'Current_Value'] = 0.0
            updated_df.loc[index, 'Day_Change_Stock_Percent'] = 0.0
            updated_df.loc[index, 'Profit_Loss'] = -current_investment_value
            profit_loss_percent_calculated = -100.0 if current_investment_value != 0 else 0.0
            updated_df.loc[index, 'Profit_Loss_Percent'] = profit_loss_percent_calculated
            current_signal = "⚠️ Price N/A"

        ath_correction_value = 0.0
        if ath is not None:
            updated_df.loc[index, 'ATH'] = ath
            if cmp is not None and ath != 0:
                ath_correction_value = ((cmp - ath) / ath) * 100
                updated_df.loc[index, 'ATH_Correction_Percent'] = ath_correction_value
            else:
                updated_df.loc[index, 'ATH_Correction_Percent'] = 0.0
        else:
            updated_df.loc[index, 'ATH'] = 0.0
            updated_df.loc[index, 'ATH_Correction_Percent'] = 0.0

        if cmp is not None:
            buy_trigger = ATH_BUY_TRIGGERS.get(category, ATH_BUY_TRIGGERS["Unknown"])
            sell_trigger = PROFIT_SELL_TRIGGERS.get(category, PROFIT_SELL_TRIGGERS["Unknown"])
            if ath is not None and ath_correction_value <= buy_trigger :
                current_signal = "⬇️ Potential Buy?"
            elif profit_loss_percent_calculated >= sell_trigger:
                current_signal = "↗️ Consider Profit?"
        updated_df.loc[index, 'Signal'] = current_signal
        current_val_for_total = updated_df.loc[index, 'Current_Value']
        total_current_value_portfolio += float(current_val_for_total) if pd.notnull(current_val_for_total) else 0.0

    if total_current_value_portfolio > 0.000001:
        updated_df['Percent_of_Portfolio'] = (updated_df['Current_Value'].astype(float) / total_current_value_portfolio) * 100
    else:
        updated_df['Percent_of_Portfolio'] = 0.0

    overall_pl_portfolio = total_current_value_portfolio - total_investment_value_portfolio
    overall_pl_percent_portfolio = (overall_pl_portfolio / total_investment_value_portfolio) * 100 if total_investment_value_portfolio != 0 else 0.0
    
    previous_day_total_value_for_pct_calc = total_current_value_portfolio - total_day_change_value_portfolio
    day_portfolio_change_percent = (total_day_change_value_portfolio / previous_day_total_value_for_pct_calc) * 100 if previous_day_total_value_for_pct_calc != 0 else 0.0

    return updated_df, {"total_invested": total_investment_value_portfolio, "total_current_value": total_current_value_portfolio, "overall_pl": overall_pl_portfolio, "overall_pl_percent": overall_pl_percent_portfolio, "day_portfolio_change_value": total_day_change_value_portfolio, "day_portfolio_change_percent": day_portfolio_change_percent}

def _format_numeric_value_for_table(x_val, format_str, col_name_for_fmt):
    if pd.isnull(x_val):
        return 'N/A' if '%' not in format_str else ('0.00%' if col_name_for_fmt != 'Quantity' else '0.000')
    try:
        return format_str.format(float(x_val))
    except (ValueError, TypeError):
        return str(x_val) if x_val is not None else ('N/A' if '%' not in format_str else '0.00%')

def get_financial_year_dates(fy_option_input):
    today = datetime.now().date()
    current_month = today.month
    current_calendar_year = today.year
    start_year, end_year = 0, 0
    display_label = "All Time"

    if isinstance(fy_option_input, str):
        if fy_option_input == "Current FY":
            if current_month >= 4: 
                start_year = current_calendar_year
            else: 
                start_year = current_calendar_year - 1
            end_year = start_year + 1
            display_label = f"Current FY ({start_year}-{str(end_year)[-2:]})"
        elif fy_option_input == "Previous FY":
            if current_month >= 4:
                start_year = current_calendar_year - 1
            else:
                start_year = current_calendar_year - 2
            end_year = start_year + 1
            display_label = f"Previous FY ({start_year}-{str(end_year)[-2:]})"
        elif fy_option_input == "All Time":
            return None, None, "All Time"
        else: 
            return None, None, "All Time"
    elif isinstance(fy_option_input, (int, float)): 
        try:
            start_year = int(fy_option_input)
            end_year = start_year + 1
            display_label = f"FY {start_year}-{str(end_year)[-2:]}"
        except ValueError:
            return None, None, "All Time" 
    else: 
        return None, None, "All Time"

    fy_start_date = pd.to_datetime(date(start_year, 4, 1))
    fy_end_date = pd.to_datetime(date(end_year, 3, 31))
    return fy_start_date, fy_end_date, display_label

@st.cache_data(ttl=6*60*60) 
def calculate_received_dividends_for_stock(ticker_symbol, purchase_date_dt, quantity, 
                                           target_fy_start_date=None, target_fy_end_date=None):
    if pd.isna(purchase_date_dt) or quantity == 0:
        return 0.0, [] 

    total_dividends_for_stock = 0.0
    dividend_details_list = []
    
    try:
        stock = yf.Ticker(ticker_symbol)
        dividends_history = stock.dividends 
        
        if dividends_history is None or dividends_history.empty:
            return 0.0, []
        
        purchase_date_ts = pd.to_datetime(purchase_date_dt)
        purchase_date_naive = purchase_date_ts.tz_localize(None) if purchase_date_ts.tzinfo is not None else purchase_date_ts

        for ex_div_date_ts, amount_per_share in dividends_history.items():
            ex_div_date_dt = pd.to_datetime(ex_div_date_ts)
            ex_div_date_naive = ex_div_date_dt.tz_localize(None) if ex_div_date_dt.tzinfo is not None else ex_div_date_dt

            condition1_met = ex_div_date_naive.date() >= purchase_date_naive.date()
            condition2_met = True 
            if target_fy_start_date is not None and target_fy_end_date is not None:
                condition2_met = (target_fy_start_date.date() <= ex_div_date_naive.date() <= target_fy_end_date.date())
            
            if condition1_met:
                if not (target_fy_start_date and target_fy_end_date) or condition2_met: 
                    dividend_received = amount_per_share * quantity
                    total_dividends_for_stock += dividend_received
                    dividend_details_list.append({
                        "Ex-Dividend Date": ex_div_date_naive.strftime('%Y-%m-%d'),
                        "Amount per Share": amount_per_share,
                        "Total Received": dividend_received
                    })
        return total_dividends_for_stock, dividend_details_list
    except Exception:
        return 0.0, []

# --- UI Rendering Functions ---
def render_add_stock_form(db):
    st.subheader("💵 Add Single Stock")
    with st.form("add_stock_form_portfolio_page_v14", clear_on_submit=True): 
        ticker = st.text_input("Ticker Symbol (e.g., RELIANCE.NS)", key="portfolio_add_ticker_v14")
        company_name = st.text_input("Company Name (Optional, will attempt to fetch)", key="portfolio_add_company_name_v14")
        purchase_date_input = st.date_input("Purchase Date", value=pd.to_datetime('today').date(), key="portfolio_add_purchase_date_v14", max_value=pd.to_datetime('today').date())
        purchase_price = st.number_input("Purchase Price", min_value=0.01, format="%.2f", step=0.01, key="portfolio_add_purchase_price_v14")
        quantity = st.number_input("Quantity", min_value=0.001, step=0.001, format="%.3f", key="portfolio_add_quantity_v14")
        notes = st.text_area("Notes/Investment Rationale (Optional)", key="portfolio_add_notes_v14")
        submitted = st.form_submit_button("Add Stock to Portfolio")
        if submitted:
            if not ticker or purchase_price <= 0 or quantity <= 0:
                st.error("Ticker, valid Purchase Price (>0), and Quantity (>0) are required.")
                return
            final_company_name = company_name.strip() if company_name else ""
            sector_val = "Unknown"
            try:
                stock_info_data = yf.Ticker(ticker.strip()).info
                if not final_company_name:
                    final_company_name = safe_get(stock_info_data, 'longName', ticker.strip().upper())
                sector_val = safe_get(stock_info_data, 'sector', 'Unknown')
            except Exception:
                if not final_company_name:
                    final_company_name = ticker.strip().upper()
            new_stock_data = {col: None for col in DEFAULT_PORTFOLIO_COLS}
            new_stock_data.update({
                "Ticker": ticker.strip().upper(), "Company_Name": final_company_name,
                "Purchase_Date": pd.to_datetime(purchase_date_input),
                "Purchase_Price": purchase_price, "Quantity": quantity,
                "Notes": notes, "Investment_Value": quantity * purchase_price,
                "Sector": sector_val if sector_val else "Unknown",
                "CMP": 0.0, "MarketCap": 0.0, "Category": "Unknown", "ATH": 0.0,
                "ATH_Correction_Percent": 0.0, "Signal": "⏳ Hold", "Current_Value": 0.0,
                "Profit_Loss": 0.0, "Profit_Loss_Percent": 0.0, "Day_Change_Stock_Percent": 0.0,
                "Percent_of_Portfolio": 0.0, "Stop_Loss": None, "Target_Price": None
            })
            if db:
                success, error_msg = add_stock_to_firestore(db, new_stock_data)
                if success:
                    st.success(f"{final_company_name} ({ticker.strip().upper()}) added.")
                    st.session_state.portfolio_df_loaded_from_firestore = False
                    for key_to_del in ['calculated_portfolio_df', 'calculated_summary_metrics']:
                        if key_to_del in st.session_state: del st.session_state[key_to_del]
                    calculate_received_dividends_for_stock.clear()
                    load_portfolio_value_history_from_firestore.clear() # Clear history cache
                    st.rerun()
                else: st.error(f"Failed to add stock to Firestore: {error_msg}")
            else: st.error("DB connection not available.")

def render_csv_upload_section(db):
    st.subheader("📄 Upload Portfolio from CSV")
    uploaded_file = st.file_uploader("Choose a CSV file (Columns: Ticker, Quantity, Purchase_Price, [Purchase_Date], [Company_Name], [Notes], [Sector])", type="csv", key="portfolio_csv_uploader_v14") 
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data (first 5 rows):")
            st.dataframe(df_upload.head())
            column_mappings = {
                "ticker": ["ticker", "tickers", "symbol", "stock symbol"],
                "quantity": ["quantity", "qty", "units"],
                "price": ["buying price", "purchase price", "price", "buy price", "avg price", "Purchase_Price"],
                "date": ["purchase date", "date", "buy date", "transaction date", "Purchase_Date"],
                "name": ["company name", "name", "company", "stock name", "Company_Name"],
                "notes": ["notes", "note", "rationale", "comment", "Notes"],
                "sector": ["sector", "industry_group", "industry", "category", "Sector"]
            }
            actual_cols = {key: None for key in column_mappings}
            df_cols_lower = [col.lower().strip().replace("_", " ") for col in df_upload.columns]
            for target_col, potential_names in column_mappings.items():
                for p_name in potential_names:
                    if p_name.lower() in df_cols_lower:
                        actual_cols[target_col] = df_upload.columns[df_cols_lower.index(p_name.lower())]
                        break
            if not actual_cols["ticker"] or not actual_cols["quantity"] or not actual_cols["price"]:
                st.error("CSV must contain columns for Ticker, Quantity, and Purchase Price. Please check column names."); return
            
            if st.button("Process and Add Stocks from CSV", key="process_csv_button_v14"): 
                with st.spinner("Processing CSV and adding stocks..."):
                    success_count = 0; failure_count = 0
                    for index, row_data in df_upload.iterrows():
                        ticker = str(row_data.get(actual_cols["ticker"], "")).strip().upper()
                        quantity = safe_to_float(row_data.get(actual_cols["quantity"]))
                        buying_price = safe_to_float(row_data.get(actual_cols["price"]))
                        purchase_date_val = date.today()
                        if actual_cols["date"] and pd.notnull(row_data.get(actual_cols["date"])):
                            try: purchase_date_val = pd.to_datetime(row_data[actual_cols["date"]]).date()
                            except: st.warning(f"Could not parse date for {ticker} (row {index+2}), using today.", icon="⚠️")

                        company_name_val = ticker
                        if actual_cols["name"] and pd.notnull(row_data.get(actual_cols["name"])):
                            company_name_val = str(row_data[actual_cols["name"]]).strip()

                        sector_val = "Unknown"
                        if actual_cols["sector"] and pd.notnull(row_data.get(actual_cols["sector"])):
                            sector_val = str(row_data[actual_cols["sector"]]).strip()
                        else: 
                            try:
                                stock_info_csv = yf.Ticker(ticker).info
                                if company_name_val == ticker: 
                                    company_name_val = safe_get(stock_info_csv, 'longName', ticker)
                                sector_val = safe_get(stock_info_csv, 'sector', 'Unknown')
                            except: pass

                        notes_val = str(row_data.get(actual_cols["notes"],"")).strip() if actual_cols["notes"] and pd.notnull(row_data.get(actual_cols["notes"])) else ""

                        if not ticker or quantity is None or quantity <= 0 or buying_price is None or buying_price <= 0:
                            st.warning(f"Skipping row {index+2}: Invalid data for Ticker '{ticker}' (Qty: {quantity}, Price: {buying_price})."); failure_count += 1; continue
                        new_stock_data = {col: None for col in DEFAULT_PORTFOLIO_COLS}
                        new_stock_data.update({
                            "Ticker": ticker, "Company_Name": company_name_val,
                            "Purchase_Date": pd.to_datetime(purchase_date_val),
                            "Purchase_Price": buying_price, "Quantity": quantity,
                            "Notes": notes_val, "Investment_Value": quantity * buying_price,
                            "Sector": sector_val if sector_val else "Unknown",
                            "CMP": 0.0, "MarketCap": 0.0, "Category": "Unknown", "ATH": 0.0,
                            "ATH_Correction_Percent": 0.0, "Signal": "⏳ Hold", "Current_Value": 0.0,
                            "Profit_Loss": 0.0, "Profit_Loss_Percent": 0.0, "Day_Change_Stock_Percent": 0.0,
                            "Percent_of_Portfolio": 0.0, "Stop_Loss": None, "Target_Price": None
                        })
                        if db:
                            added, error_msg = add_stock_to_firestore(db, new_stock_data)
                            if added: success_count += 1
                            else: failure_count += 1; st.error(f"Failed to add {ticker} (row {index+2}) from CSV: {error_msg}")
                        else: failure_count += 1; st.error("DB connection not available for CSV upload."); break
                    st.success(f"Processed CSV: {success_count} stocks added, {failure_count} stocks failed.")
                    if success_count > 0:
                        st.session_state.portfolio_df_loaded_from_firestore = False
                        for key_to_del in ['calculated_portfolio_df', 'calculated_summary_metrics']:
                            if key_to_del in st.session_state: del st.session_state[key_to_del]
                        calculate_received_dividends_for_stock.clear()
                        load_portfolio_value_history_from_firestore.clear()
                        st.rerun()
        except Exception as e: st.error(f"Error processing CSV file: {e}")

def render_portfolio_holdings_table(portfolio_df_display):
    st.subheader("📈 Current Portfolio Holdings")
    if portfolio_df_display is None or portfolio_df_display.empty:
        st.info("Your portfolio is currently empty or data is being loaded. Add stocks or refresh."); return
    display_columns_ordered = [
        "Ticker", "Company_Name", "Quantity", "Purchase_Date", "Purchase_Price",
        "Investment_Value", "CMP", "Current_Value", "MarketCap", "Category", "Sector",
        "Profit_Loss", "Profit_Loss_Percent", "Day_Change_Stock_Percent",
        "ATH", "ATH_Correction_Percent", "Signal", "Percent_of_Portfolio", "Notes"
    ]
    df_view = portfolio_df_display.copy()
    if 'Purchase_Date' in df_view.columns and not df_view['Purchase_Date'].empty:
        df_view['Purchase_Date'] = pd.to_datetime(df_view['Purchase_Date'], errors='coerce')
        df_view['Purchase_Date'] = df_view['Purchase_Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) and isinstance(x, (datetime, pd.Timestamp)) else 'N/A')
    if 'MarketCap' in df_view.columns:
        df_view['MarketCap'] = df_view['MarketCap'].apply(lambda x: format_large_number(x, default='N/A') if pd.notnull(x) else 'N/A')
    
    numeric_formats = {
        "Quantity": "{:,.3f}", "Purchase_Price": "{:,.2f}", "Investment_Value": "{:,.2f}",
        "CMP": "{:,.2f}", "Current_Value": "{:,.2f}", "Profit_Loss": "{:,.2f}",
        "Profit_Loss_Percent": "{:.2f}%", "Day_Change_Stock_Percent": "{:.2f}%",
        "ATH": "{:,.2f}", "ATH_Correction_Percent": "{:.2f}%", "Percent_of_Portfolio": "{:.2f}%"
    }
    for col, fmt in numeric_formats.items():
        if col in df_view.columns:
            df_view[col] = df_view[col].apply(lambda x: _format_numeric_value_for_table(x, fmt, col))
            
    for col_name in ['Sector', 'Category', 'Signal', 'Notes', 'Company_Name', 'Ticker']:
        if col_name in df_view.columns: df_view[col_name] = df_view[col_name].fillna("N/A")
    
    final_display_cols = [col for col in display_columns_ordered if col in df_view.columns]
    st.dataframe(df_view[final_display_cols], use_container_width=True, hide_index=True)

def render_manage_holdings_section(db, portfolio_df_for_management):
    st.subheader("🗑️ Manage Individual Holdings")
    if portfolio_df_for_management is not None and not portfolio_df_for_management.empty and 'Ticker' in portfolio_df_for_management.columns:
        valid_tickers = portfolio_df_for_management['Ticker'].dropna().astype(str)
        unique_tickers = valid_tickers.unique()
        tickers_in_portfolio = [""] + sorted(list(unique_tickers))
        selected_ticker_for_removal = st.selectbox("Select Ticker to Remove:", options=tickers_in_portfolio, key="portfolio_remove_ticker_select_v12", index=0) 
        if selected_ticker_for_removal and st.button(f"Confirm Removal of {selected_ticker_for_removal}", key=f"portfolio_remove_{selected_ticker_for_removal}_btn_v12"): 
            if db:
                success, error_msg = delete_stock_from_firestore(db, selected_ticker_for_removal)
                if success:
                    st.success(f"{selected_ticker_for_removal} removed from portfolio.")
                    st.session_state.portfolio_df_loaded_from_firestore = False
                    for key_to_del in ['calculated_portfolio_df', 'calculated_summary_metrics', 'portfolio_df']:
                        if key_to_del in st.session_state: del st.session_state[key_to_del]
                    calculate_received_dividends_for_stock.clear()
                    load_portfolio_value_history_from_firestore.clear()
                    st.rerun()
                else: st.error(f"Failed to remove {selected_ticker_for_removal} from Firestore: {error_msg}")
            else: st.error("Database connection not available. Cannot remove stock.")
    else:
        st.info("No holdings available to manage, or 'Ticker' column is missing.")

def render_portfolio_summary(summary_metrics):
    st.subheader("📊 Portfolio Performance Summary")
    # ... (same as before)
    has_data_in_portfolio_df = st.session_state.get('portfolio_df') is not None and not st.session_state.get('portfolio_df').empty
    has_calculated_metrics = summary_metrics and (summary_metrics.get("total_invested") != 0 or summary_metrics.get("total_current_value") != 0)
    
    if not has_data_in_portfolio_df and not has_calculated_metrics:
        st.info("Portfolio is empty. Add stocks to see a summary."); return
    elif not has_calculated_metrics and has_data_in_portfolio_df:
        st.info("Calculating summary or data might be zero. Refresh live data if needed.")
        if not summary_metrics:
            summary_metrics = {"total_invested": 0, "total_current_value": 0, "overall_pl": 0,
                               "overall_pl_percent": 0.0, "day_portfolio_change_value": 0, "day_portfolio_change_percent": 0.0}
    cols = st.columns(4)
    cols[0].metric(label="Total Invested", value=format_large_number(summary_metrics.get("total_invested", 0)))
    cols[1].metric(label="Current Value", value=format_large_number(summary_metrics.get("total_current_value", 0)))
    
    overall_pl_val = summary_metrics.get("overall_pl", 0); overall_pl_pct = summary_metrics.get("overall_pl_percent", 0.0)
    overall_pl_color = "red" if overall_pl_val < 0 else ("green" if overall_pl_val > 0 else "gray")
    overall_pl_arrow = "🔻" if overall_pl_val < 0 else ("🔼" if overall_pl_val > 0 else "")
    overall_pl_html = f"""<div style="padding: 0.2rem 0rem;"><span style='font-size: 0.875rem; color: #808495; display: block; margin-bottom: 0.0rem;'>Overall P/L</span><span style='font-size: 1.625rem; color: {overall_pl_color}; display: inline-block; font-weight: 500;'>{format_large_number(overall_pl_val)}</span><span style='font-size: 0.875rem; color: {overall_pl_color}; margin-left: 0.3rem;'>{overall_pl_arrow} ({overall_pl_pct:.2f}%)</span></div>"""
    cols[2].markdown(overall_pl_html, unsafe_allow_html=True)
    
    day_pl_val = summary_metrics.get("day_portfolio_change_value", 0); day_pl_pct = summary_metrics.get("day_portfolio_change_percent", 0.0)
    day_pl_color = "red" if day_pl_val < 0 else ("green" if day_pl_val > 0 else "gray")
    day_pl_arrow = "🔻" if day_pl_val < 0 else ("🔼" if day_pl_val > 0 else "")
    day_pl_main_value_html = f"{format_large_number(day_pl_val)}"
    day_pl_delta_html = f"{day_pl_arrow} {day_pl_pct:.2f}% today"
    day_pl_html = f"""<div style="padding: 0.2rem 0rem;"><span style='font-size: 0.875rem; color: #808495; display: block; margin-bottom: 0.0rem;'>Day's P/L</span><span style='font-size: 1.625rem; color: {day_pl_color}; display: inline-block; font-weight: 500;'>{day_pl_main_value_html}</span><span style='font-size: 0.875rem; color: {day_pl_color}; margin-left: 0.3rem;'>{day_pl_delta_html}</span></div>"""
    cols[3].markdown(day_pl_html, unsafe_allow_html=True)

def render_dividend_tracker(portfolio_df):
    # ... (This function remains the same, using Financial Year logic)
    st.subheader("💵 Dividend Income Tracker")
    if portfolio_df is None or portfolio_df.empty:
        st.info("Portfolio is empty. Add stocks to track dividends.")
        return

    current_calendar_year = datetime.now().year
    
    _, _, current_fy_label_full = get_financial_year_dates("Current FY")
    _, _, prev_fy_label_full = get_financial_year_dates("Previous FY")

    fy_options_display = ["All Time", current_fy_label_full, prev_fy_label_full, "Custom FY"]
    
    if 'dividend_fy_option_label_v2' not in st.session_state:
        st.session_state.dividend_fy_option_label_v2 = "All Time"
    if 'dividend_custom_fy_start_year_input_v2' not in st.session_state:
        st.session_state.dividend_custom_fy_start_year_input_v2 = current_calendar_year -1 

    selected_fy_display_label = st.selectbox(
        "Select Dividend Financial Year:",
        options=fy_options_display,
        index=fy_options_display.index(st.session_state.dividend_fy_option_label_v2),
        key="dividend_fy_select_sb_v2"
    )
    st.session_state.dividend_fy_option_label_v2 = selected_fy_display_label

    target_fy_start_date, target_fy_end_date, display_label_for_summary = None, None, "All Time"

    if selected_fy_display_label == "All Time":
        target_fy_start_date, target_fy_end_date, display_label_for_summary = None, None, "All Time"
    elif selected_fy_display_label == current_fy_label_full:
        target_fy_start_date, target_fy_end_date, display_label_for_summary = get_financial_year_dates("Current FY")
    elif selected_fy_display_label == prev_fy_label_full:
        target_fy_start_date, target_fy_end_date, display_label_for_summary = get_financial_year_dates("Previous FY")
    elif selected_fy_display_label == "Custom FY":
        custom_start_year = st.number_input(
            "Enter Start Year of Custom FY (e.g., 2022 for FY 2022-23):",
            min_value=1990,
            max_value=current_calendar_year + 5, 
            value=st.session_state.dividend_custom_fy_start_year_input_v2,
            step=1,
            key="dividend_custom_fy_input_field_v2"
        )
        st.session_state.dividend_custom_fy_start_year_input_v2 = custom_start_year
        
        if custom_start_year:
            target_fy_start_date, target_fy_end_date, display_label_for_summary = get_financial_year_dates(custom_start_year)
            if target_fy_start_date is None:
                st.warning("Invalid custom financial year input. Showing 'All Time'.")
                target_fy_start_date, target_fy_end_date, display_label_for_summary = None, None, "All Time (Custom FY Invalid)"
        else:
            st.warning("Please enter a start year for 'Custom FY'. Showing 'All Time'.")
            target_fy_start_date, target_fy_end_date, display_label_for_summary = None, None, "All Time (Custom FY not set)"

    all_dividends_data = []
    grand_total_dividends = 0.0

    required_cols = ["Ticker", "Purchase_Date", "Quantity"]
    if not all(col in portfolio_df.columns for col in required_cols):
        missing_cols_str = ", ".join([col for col in required_cols if col not in portfolio_df.columns])
        st.error(f"Portfolio data is missing required columns for dividend tracking: {missing_cols_str}")
        return

    with st.spinner(f"Calculating received dividends for {display_label_for_summary}..."):
        for index, row in portfolio_df.iterrows():
            ticker = row["Ticker"]
            purchase_date = pd.to_datetime(row["Purchase_Date"], errors='coerce') 
            quantity = row["Quantity"]
            company_name = row.get("Company_Name", ticker)

            if pd.isna(purchase_date) or quantity == 0:
                continue
            
            received, details = calculate_received_dividends_for_stock(
                ticker, purchase_date, quantity, 
                target_fy_start_date=target_fy_start_date, 
                target_fy_end_date=target_fy_end_date
            )
            if received > 0:
                all_dividends_data.append({
                    "Ticker": ticker,
                    "Company": company_name,
                    "Total Received (₹)": received,
                    "Purchase Date": purchase_date.strftime('%Y-%m-%d'),
                    "Quantity": quantity,
                    "Details": details 
                })
            grand_total_dividends += received
    
    if not all_dividends_data and target_fy_start_date: 
        st.info(f"No dividends found for {display_label_for_summary} for the current holdings, or dividend data not available for this period.")
    elif not all_dividends_data and target_fy_start_date is None:
        st.info("No dividends received for the current holdings (All Time), or dividend data not available from yfinance.")

    st.metric(f"Total Dividends Received ({display_label_for_summary})", f"₹{grand_total_dividends:,.2f}")

    if all_dividends_data:
        summary_df = pd.DataFrame(all_dividends_data)[["Ticker", "Company", "Total Received (₹)"]]
        summary_df = summary_df.sort_values(by="Total Received (₹)", ascending=False).reset_index(drop=True)
        
        st.markdown(f"##### Dividends by Stock ({display_label_for_summary}):")
        st.dataframe(summary_df, use_container_width=True, hide_index=True, 
                       column_config={"Total Received (₹)": st.column_config.NumberColumn(format="₹ %.2f")})

        with st.expander(f"View Detailed Dividend History ({display_label_for_summary})"):
            if not all_dividends_data:
                st.caption("No detailed dividend history to show.")
            for item in all_dividends_data:
                if item["Details"]:
                    st.markdown(f"**{item['Company']} ({item['Ticker']}) - Total for {display_label_for_summary}: ₹{item['Total Received (₹)']:,.2f}** (Qty: {item['Quantity']:.3f} since {item['Purchase Date']})")
                    details_df = pd.DataFrame(item["Details"])
                    st.dataframe(details_df, hide_index=True, use_container_width=True,
                                   column_config={
                                       "Total Received": st.column_config.NumberColumn(format="₹ %.2f"),
                                       "Amount per Share": st.column_config.NumberColumn(format="%.4f")
                                   })

def render_portfolio_visualizations(db, display_df): # db passed for portfolio value history
    st.subheader("✨ Portfolio Visualizations")

    if display_df is None or display_df.empty:
        st.info("Not enough data for visualizations. Add stocks and refresh.")
        return

    if 'Current_Value' in display_df.columns:
        display_df['Current_Value'] = pd.to_numeric(display_df['Current_Value'], errors='coerce').fillna(0.0)
    else:
        st.warning("Critical: 'Current_Value' column is missing. Some allocation visualizations may not render.", icon="⚠️")

    can_show_allocation_charts = 'Current_Value' in display_df.columns and display_df['Current_Value'].sum() > 0.01

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        # ... (Allocation by Stock Treemap - same as before) ...
        st.markdown("##### Allocation by Stock (Treemap)")
        if can_show_allocation_charts and 'Company_Name' in display_df.columns:
            stock_alloc_df = display_df.groupby("Company_Name")['Current_Value'].sum().reset_index()
            stock_alloc_df = stock_alloc_df[stock_alloc_df['Current_Value'] > 0.01]
            if not stock_alloc_df.empty:
                fig_stock_treemap = px.treemap(stock_alloc_df, path=[px.Constant("Portfolio"), 'Company_Name'], values='Current_Value', title='Stock Allocation by Current Value', height=450, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_stock_treemap.update_traces(textinfo='label+percent parent', hovertemplate="<b>%{label}</b><br>Allocation: %{value:,.2f}<br>Percentage: %{percentParent:.2%}<extra></extra>")
                fig_stock_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_stock_treemap, use_container_width=True)
            else: st.caption("No significant stock values for treemap.")
        elif not can_show_allocation_charts:
            st.caption("Current portfolio value is zero or negligible; cannot display stock allocation.")
        else: st.caption("Company Name or Current Value data missing for stock allocation treemap.")

    with viz_col2:
        # ... (Allocation by Sector Pie Chart - same as before) ...
        st.markdown("##### Allocation by Sector")
        if can_show_allocation_charts and 'Sector' in display_df.columns:
            display_df_sector = display_df.copy()
            display_df_sector['Sector'] = display_df_sector['Sector'].fillna("Unknown")
            sector_alloc_df = display_df_sector.groupby("Sector")['Current_Value'].sum().reset_index()
            sector_alloc_df = sector_alloc_df[sector_alloc_df['Current_Value'] > 0.01]
            if not sector_alloc_df.empty:
                fig_sector = px.pie(sector_alloc_df, values='Current_Value', names='Sector', title='by Sector', hole=0.45, height=450)
                fig_sector.update_traces(textfont_size=10, textposition='inside', textinfo='percent+label', insidetextorientation='radial')
                fig_sector.update_layout(showlegend=True, legend_traceorder="reversed", legend_orientation="v", legend_y=0.5, legend_x=1.05, margin=dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_sector, use_container_width=True)
            else: st.caption("No significant sector values for chart.")
        elif not can_show_allocation_charts:
            st.caption("Current portfolio value is zero or negligible; cannot display sector allocation.")
        else: st.caption("Sector or Current Value data missing for sector allocation chart.")

    st.markdown("---")
    viz_col3, viz_col4 = st.columns(2)
    with viz_col3:
        # ... (Allocation by Market Cap Pie Chart - same as before) ...
        st.markdown("##### Allocation by Market Cap")
        if can_show_allocation_charts and 'Category' in display_df.columns:
            display_df_cat = display_df.copy()
            display_df_cat['Category'] = display_df_cat['Category'].fillna("Unknown")
            mcap_alloc_df = display_df_cat.groupby("Category")['Current_Value'].sum().reset_index()
            mcap_alloc_df = mcap_alloc_df[mcap_alloc_df['Current_Value'] > 0.01]
            if not mcap_alloc_df.empty:
                fig_mcap = px.pie(mcap_alloc_df, values='Current_Value', names='Category', title='by Market Cap Category', hole=0.45, height=450)
                fig_mcap.update_traces(textfont_size=10, textposition='inside', textinfo='percent+label', insidetextorientation='radial')
                fig_mcap.update_layout(showlegend=True, legend_traceorder="reversed", legend_orientation="v", legend_y=0.5, legend_x=1.05, margin=dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig_mcap, use_container_width=True)
            else: st.caption("No significant market cap category values for chart.")
        elif not can_show_allocation_charts:
            st.caption("Current portfolio value is zero or negligible; cannot display market cap allocation.")
        else: st.caption("Category (Market Cap) or Current Value data missing for market cap allocation chart.")


    with viz_col4: # Profit/Loss by Stock (Styled Table)
        st.markdown("##### Profit/Loss by Stock")
        if not display_df.empty and \
           all(col in display_df.columns for col in ['Profit_Loss', 'Company_Name', 'Profit_Loss_Percent']):
            
            pl_table_df = display_df[['Company_Name', 'Ticker', 'Profit_Loss', 'Profit_Loss_Percent']].copy()
            
            pl_table_df['Profit_Loss'] = pd.to_numeric(pl_table_df['Profit_Loss'], errors='coerce').fillna(0.0)
            pl_table_df['Profit_Loss_Percent'] = pd.to_numeric(pl_table_df['Profit_Loss_Percent'], errors='coerce').fillna(0.0)

            pl_table_df = pl_table_df.sort_values(by='Profit_Loss', ascending=False).reset_index(drop=True)

            pl_table_df.rename(columns={
                'Company_Name': 'Stock',
                'Profit_Loss': 'P/L (₹)',
                'Profit_Loss_Percent': 'P/L (%)'
            }, inplace=True)

            def color_pl_value(val): 
                if pd.isna(val): return ''
                color = 'darkgreen' if val > 0 else 'maroon' if val < 0 else 'dimgray'
                return f'color: {color}; font-weight: bold;' if val !=0 else 'color: dimgray;'
            
            def color_pl_percent(val):
                if pd.isna(val): return ''
                color = 'green' if val > 0 else 'red' if val < 0 else 'grey'
                return f'color: {color};'

            display_columns_for_table = ['Stock', 'P/L (₹)', 'P/L (%)']
            styled_df = pl_table_df[display_columns_for_table].style \
                .applymap(color_pl_value, subset=['P/L (₹)']) \
                .applymap(color_pl_percent, subset=['P/L (%)']) \
                .format({
                    'P/L (₹)': "₹{:,.2f}",
                    'P/L (%)': "{:.2f}%"
                })
            
            num_rows = len(pl_table_df)
            table_height = min((num_rows + 1) * 35 + 3, 425) 

            st.dataframe(styled_df, use_container_width=True, height=table_height) 

        elif display_df.empty:
            st.caption("Portfolio is empty.")
        else:
            st.caption("Required P/L data (Profit_Loss, Company_Name, Profit_Loss_Percent) missing for this table.")

    st.markdown("---")
    st.markdown("### Portfolio Value Trend Over Time")
    history_value_df, error_msg = load_portfolio_value_history_from_firestore(db, USER_ID_PLACEHOLDER)
    
    if error_msg:
        st.error(f"Could not load portfolio value history: {error_msg}")
    
    if not history_value_df.empty:
        fig_value_trend = px.line(history_value_df, x='Date', y='TotalValue', markers=True,
                                  title="Logged Portfolio Value Over Time")
        fig_value_trend.update_layout(yaxis_title="Total Portfolio Value (₹)")
        st.plotly_chart(fig_value_trend, use_container_width=True)
    else:
        st.info("No portfolio value history logged yet. Use the 'Log Value' button or ensure automated logger is running.")
    st.markdown("---") 

    with st.expander("🔬 Advanced Portfolio Analysis (Experimental)", expanded=False):
        st.markdown("### Stock Correlation Matrix")
        if display_df.empty or 'Ticker' not in display_df.columns or len(display_df['Ticker'].unique()) < 2:
            st.info("Need at least two stocks in the portfolio to calculate a stock correlation matrix.")
        else:
            tickers_for_stock_corr = display_df['Ticker'].unique().tolist()
            period_options_stock_corr = {
                "Last 3 Months": "3mo", "Last 6 Months": "6mo",
                "Last 1 Year": "1y", "Last 2 Years": "2y"
            }
            if 'stock_corr_period_label' not in st.session_state:
                st.session_state.stock_corr_period_label = "Last 1 Year"

            selected_period_label_stock_corr = st.selectbox(
                "Select Period for Stock Correlation:",
                options=list(period_options_stock_corr.keys()),
                index=list(period_options_stock_corr.keys()).index(st.session_state.stock_corr_period_label),
                key="stock_corr_matrix_period_select_v14" 
            )
            st.session_state.stock_corr_period_label = selected_period_label_stock_corr
            yf_period_stock_corr = period_options_stock_corr[selected_period_label_stock_corr]

            if st.button("📊 Generate Stock Correlation Matrix", key="gen_stock_corr_matrix_btn_v14"): 
                with st.spinner(f"Fetching data for stock correlation ({selected_period_label_stock_corr})..."):
                    all_price_data_stock = {}
                    valid_tickers_for_stock_matrix = []
                    for ticker_sc in tickers_for_stock_corr:
                        hist_df_stock = get_stock_history(ticker_sc, period=yf_period_stock_corr, interval="1d")
                        if hist_df_stock is not None and not hist_df_stock.empty and 'Close' in hist_df_stock.columns and not hist_df_stock['Close'].isnull().all():
                            all_price_data_stock[ticker_sc] = hist_df_stock['Close']
                            valid_tickers_for_stock_matrix.append(ticker_sc)
                        else: st.warning(f"Stock Corr: No valid data for {ticker_sc}.", icon="⚠️")
                    
                    if len(valid_tickers_for_stock_matrix) < 2:
                        st.error("Not enough stocks with valid historical data for stock correlation matrix.")
                    else:
                        price_matrix_df_stock = pd.DataFrame(all_price_data_stock).ffill().bfill()
                        price_matrix_df_stock.dropna(axis=1, how='all', inplace=True)
                        price_matrix_df_stock.dropna(axis=0, how='all', inplace=True)

                        if price_matrix_df_stock.shape[1] < 2 or price_matrix_df_stock.empty:
                            st.error("Not enough valid stock data after cleaning for stock correlation.")
                        else:
                            stock_returns_df_corr = price_matrix_df_stock.pct_change().dropna(how='all', axis=0)
                            if stock_returns_df_corr.empty or len(stock_returns_df_corr) < 2 or stock_returns_df_corr.shape[1] < 2:
                                st.error("Not enough return data for stock correlation. Try a longer period.")
                            else:
                                stock_correlation_matrix = stock_returns_df_corr.corr()
                                ticker_to_sector_map_for_stock_corr = pd.Series(display_df.Sector.values, index=display_df.Ticker).to_dict()
                                new_labels_for_stock_corr_matrix = [f"{t} ({ticker_to_sector_map_for_stock_corr.get(t, 'N/A')})" for t in stock_correlation_matrix.columns]
                                stock_corr_matrix_display = stock_correlation_matrix.copy()
                                stock_corr_matrix_display.columns = new_labels_for_stock_corr_matrix
                                stock_corr_matrix_display.index = new_labels_for_stock_corr_matrix

                                fig_stock_corr_display = px.imshow(stock_corr_matrix_display, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title=f"Stock Returns Correlation Matrix ({selected_period_label_stock_corr})")
                                fig_stock_corr_display.update_xaxes(side="bottom", tickangle=-45); fig_stock_corr_display.update_yaxes(tickmode='linear')
                                fig_stock_corr_display.update_traces(hovertemplate="<b>Stock 1:</b> %{y}<br><b>Stock 2:</b> %{x}<br><b>Correlation:</b> %{z:.2f}<extra></extra>")
                                num_items_stock_corr_viz = len(stock_corr_matrix_display.columns); plot_size_stock_corr_viz = max(450, 35 * num_items_stock_corr_viz + 150)
                                fig_stock_corr_display.update_layout(height=plot_size_stock_corr_viz, width=plot_size_stock_corr_viz + 100, margin=dict(l=180, r=50, b=180, t=100), coloraxis_colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["-1 (Neg)", "-0.5", "0 (None)", "0.5", "1 (Pos)"]))
                                st.plotly_chart(fig_stock_corr_display, use_container_width=False)
                                st.caption("Correlation of daily returns. Red: positive, Blue: negative, White: no correlation.")
        
        st.markdown("---")
        st.markdown("### Sector-Level Correlation Matrix")
        if display_df.empty or 'Ticker' not in display_df.columns or 'Sector' not in display_df.columns or display_df['Sector'].nunique() < 2:
            st.info("Need at least two unique sectors with stock data to calculate a sector correlation matrix.")
        else:
            if st.button("📊 Generate Sector Correlation Matrix", key="gen_sector_corr_matrix_btn_v14"): 
                selected_period_label_sector_corr = st.session_state.stock_corr_period_label 
                yf_period_sector_corr = period_options_stock_corr.get(selected_period_label_sector_corr, "1y") 

                with st.spinner(f"Calculating sector correlations for {selected_period_label_sector_corr}..."): 
                    all_price_data_sector = {}
                    tickers_for_sector_corr_calc = display_df['Ticker'].unique().tolist()
                    
                    for ticker_sac in tickers_for_sector_corr_calc:
                        hist_df_sac = get_stock_history(ticker_sac, period=yf_period_sector_corr, interval="1d") 
                        if hist_df_sac is not None and not hist_df_sac.empty and 'Close' in hist_df_sac.columns and not hist_df_sac['Close'].isnull().all():
                            all_price_data_sector[ticker_sac] = hist_df_sac['Close']
                    
                    if len(all_price_data_sector) < 1:
                        st.error("Not enough stocks with valid historical data for sector correlation.")
                    else:
                        price_matrix_df_sector_calc = pd.DataFrame(all_price_data_sector).ffill().bfill()
                        price_matrix_df_sector_calc.dropna(axis=1, how='all', inplace=True)
                        price_matrix_df_sector_calc.dropna(axis=0, how='all', inplace=True)

                        if price_matrix_df_sector_calc.shape[1] < 1 or price_matrix_df_sector_calc.empty:
                            st.error("No valid stock price data remains after cleaning for sector correlation.")
                        else:
                            stock_returns_df_sector_calc = price_matrix_df_sector_calc.pct_change().dropna(how='all', axis=0)
                            if stock_returns_df_sector_calc.empty or stock_returns_df_sector_calc.shape[1] < 1:
                                st.error("Could not calculate stock returns for sector analysis.")
                            else:
                                ticker_to_sector_map_sa_calc = pd.Series(display_df.Sector.values, index=display_df.Ticker).to_dict()
                                stock_returns_with_sectors_sa_calc = stock_returns_df_sector_calc.T
                                stock_returns_with_sectors_sa_calc['Sector'] = stock_returns_with_sectors_sa_calc.index.map(ticker_to_sector_map_sa_calc)
                                stock_returns_with_sectors_sa_calc.dropna(subset=['Sector'], inplace=True)

                                if stock_returns_with_sectors_sa_calc.empty or stock_returns_with_sectors_sa_calc['Sector'].nunique() < 2:
                                    st.error("Not enough unique sectors or valid returns to calculate sector correlations.")
                                else:
                                    sector_daily_returns_sa_calc = stock_returns_with_sectors_sa_calc.groupby('Sector').mean(numeric_only=True) 
                                    sector_daily_returns_transposed_sa_calc = sector_daily_returns_sa_calc.T

                                    if sector_daily_returns_transposed_sa_calc.shape[1] < 2:
                                        st.error("Fewer than 2 sectors with aggregated data, cannot compute sector correlation.")
                                    else:
                                        sector_correlation_matrix_final = sector_daily_returns_transposed_sa_calc.corr()
                                        fig_sector_corr_display = px.imshow(sector_correlation_matrix_final, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title=f"Sector Returns Correlation Matrix ({selected_period_label_sector_corr})")
                                        fig_sector_corr_display.update_xaxes(side="bottom", tickangle=-45); fig_sector_corr_display.update_yaxes(tickmode='linear')
                                        num_sectors_corr_disp = len(sector_correlation_matrix_final.columns); plot_size_sector_corr_disp = max(400, 40 * num_sectors_corr_disp + 100)
                                        fig_sector_corr_display.update_layout(height=plot_size_sector_corr_disp, width=plot_size_sector_corr_disp + 100, margin=dict(l=120, r=50, b=120, t=100), coloraxis_colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=["-1 (Neg)", "-0.5", "0 (None)", "0.5", "1 (Pos)"]))
                                        st.plotly_chart(fig_sector_corr_display, use_container_width=False)
                                        st.caption("Correlation between average daily returns of sectors in your portfolio.")
        st.markdown("---")
        st.markdown("### Placeholder for Other Advanced Visualizations")
        st.info("More advanced analytics (e.g., risk-return scatter plot, efficient frontier) can be added here.")
    # REMOVED Top/Bottom Performers section and Other Future Enhancements text from here

# Ensure render_portfolio_page definition is complete and correct
# (The following is the complete render_portfolio_page from last good state)
def render_portfolio_page(db):
    st.title("My Portfolio 📊🚦")

    default_portfolio_df = lambda: pd.DataFrame(columns=DEFAULT_PORTFOLIO_COLS)
    default_summary_metrics = lambda: {"total_invested": 0, "total_current_value": 0, "overall_pl": 0,
                                       "overall_pl_percent": 0, "day_portfolio_change_value": 0, "day_portfolio_change_percent": 0}
    if 'portfolio_df' not in st.session_state:
        st.session_state.portfolio_df = default_portfolio_df()
    if 'portfolio_df_loaded_from_firestore' not in st.session_state:
        st.session_state.portfolio_df_loaded_from_firestore = False
    if 'calculated_portfolio_df' not in st.session_state:
        st.session_state.calculated_portfolio_df = default_portfolio_df()
    if 'calculated_summary_metrics' not in st.session_state:
        st.session_state.calculated_summary_metrics = default_summary_metrics()
    
    if 'dividend_fy_option_label_v2' not in st.session_state:
        st.session_state.dividend_fy_option_label_v2 = "All Time"
    if 'dividend_custom_fy_start_year_input_v2' not in st.session_state:
        st.session_state.dividend_custom_fy_start_year_input_v2 = datetime.now().year - 1

    button_cols = st.columns([2, 2, 1]) 
    with button_cols[0]:
        if st.button("🔄 Refresh Live Portfolio Data", key="portfolio_refresh_data_main_v14", help="Fetches latest market prices and recalculates P&L."): 
            if 'calculated_portfolio_df' in st.session_state: del st.session_state['calculated_portfolio_df']
            if 'calculated_summary_metrics' in st.session_state: del st.session_state['calculated_summary_metrics']
            calculate_received_dividends_for_stock.clear() 
            load_portfolio_value_history_from_firestore.clear()
            st.rerun()
    with button_cols[1]:
        if st.button("🔁 Reload Portfolio from DB", key="portfolio_reload_from_db_v14", help="Fetches the latest saved portfolio from Firestore."): 
            st.session_state.portfolio_df_loaded_from_firestore = False
            for key_to_del in ['portfolio_df', 'calculated_portfolio_df', 'calculated_summary_metrics']:
                if key_to_del in st.session_state: del st.session_state[key_to_del]
            calculate_received_dividends_for_stock.clear() 
            load_portfolio_value_history_from_firestore.clear()
            st.rerun()
    
    current_summary_for_manual_log = st.session_state.get('calculated_summary_metrics', default_summary_metrics())
    with button_cols[2]:
        if current_summary_for_manual_log and current_summary_for_manual_log.get("total_current_value", 0) > 0:
            if st.button("📝 Log Val", key="manual_log_portfolio_value_button_v14", help="Manually log current total portfolio value for trend chart."): 
                today_str = datetime.now().strftime("%Y-%m-%d")
                success, msg = log_portfolio_value_to_firestore(
                    db, 
                    USER_ID_PLACEHOLDER,
                    today_str, 
                    current_summary_for_manual_log["total_current_value"]
                )
                if success:
                    st.toast(f"Portfolio value manually logged for {today_str}!", icon="📝")
                    load_portfolio_value_history_from_firestore.clear() 
                    st.rerun() 
                else:
                    st.error(f"Failed to log value: {msg}")
        else:
            st.write("") 

    if not st.session_state.portfolio_df_loaded_from_firestore:
        if db:
            with st.spinner("Loading portfolio from Firestore..."):
                # Ensure DEFAULT_PORTFOLIO_COLS is available to get_portfolio_from_firestore
                # This usually happens if it's a global in firebase_utils or passed as param
                df_from_db, error_msg = get_portfolio_from_firestore(db) 
                if error_msg:
                    st.error(f"Error loading portfolio from Firestore: {error_msg}")
                    st.session_state.portfolio_df = default_portfolio_df()
                else:
                    st.session_state.portfolio_df = df_from_db if (df_from_db is not None and not df_from_db.empty) else default_portfolio_df()
                st.session_state.portfolio_df_loaded_from_firestore = True
                if 'calculated_portfolio_df' in st.session_state: del st.session_state['calculated_portfolio_df']
                if 'calculated_summary_metrics' in st.session_state: del st.session_state['calculated_summary_metrics']
                st.rerun()
        else:
            st.warning("Database not connected. Portfolio operations are local.", icon="⚠️")
            if st.session_state.get('portfolio_df') is None :
                st.session_state.portfolio_df = default_portfolio_df()

    needs_calculation = False
    base_portfolio_df = st.session_state.get('portfolio_df', default_portfolio_df())
    calculated_df = st.session_state.get('calculated_portfolio_df', default_portfolio_df())

    if 'calculated_portfolio_df' not in st.session_state or calculated_df.empty:
        if not base_portfolio_df.empty:
            needs_calculation = True
    elif not base_portfolio_df.empty and (len(calculated_df) != len(base_portfolio_df) or not calculated_df['Ticker'].equals(base_portfolio_df['Ticker'])):
        needs_calculation = True
    if base_portfolio_df.empty and not calculated_df.empty :
        needs_calculation = True

    if needs_calculation:
        if not base_portfolio_df.empty:
            with st.spinner("Fetching live prices & calculating portfolio metrics..."):
                df_calc, summary_calc = calculate_portfolio_metrics(base_portfolio_df.copy()) 
                st.session_state.calculated_portfolio_df = df_calc
                st.session_state.calculated_summary_metrics = summary_calc
        else:
            st.session_state.calculated_portfolio_df = default_portfolio_df()
            st.session_state.calculated_summary_metrics = default_summary_metrics()
            
    display_df_final = st.session_state.get('calculated_portfolio_df', default_portfolio_df())
    display_summary_final = st.session_state.get('calculated_summary_metrics', default_summary_metrics()) 

    render_portfolio_summary(display_summary_final)
    st.markdown("---")
    render_portfolio_holdings_table(display_df_final)
    st.markdown("---")
    
    expand_add_form = st.session_state.get('portfolio_df', default_portfolio_df()).empty
    with st.expander("➕ Add / Upload / Manage Portfolio", expanded=expand_add_form):
        render_add_stock_form(db)
        st.markdown("---")
        render_csv_upload_section(db)
        st.markdown("---")
        df_for_management = display_df_final if not display_df_final.empty else st.session_state.get('portfolio_df', default_portfolio_df())
        render_manage_holdings_section(db, df_for_management)
    
    st.markdown("---")
    render_portfolio_visualizations(db, display_df_final) # Pass db
    
    st.markdown("---")
    render_dividend_tracker(display_df_final)

    st.markdown("---")
    st.subheader("Future: VBRS Framework Integration")
    st.caption("PE-Based Reallocation, Cash Ratio Formula, Sector Rotation.")