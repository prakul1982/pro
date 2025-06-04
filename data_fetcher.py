# dontknwo/data_fetcher.py
import yfinance as yf
import pandas as pd
# import requests # Removed: Not used in the provided functions
from bs4 import BeautifulSoup
# from io import StringIO # Removed: Not used
from datetime import datetime, timedelta, date
import re
import time # For adding delays if needed
import streamlit as st

# Selenium Imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    # from webdriver_manager.chrome import ChromeDriverManager # Uncomment if you want to use it
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("WARNING: Selenium libraries not found. Web scraping features requiring Selenium will be disabled.")


MONTH_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def get_stock_info(ticker_symbol):
    st.write(f"--- Debug: `get_stock_info` called for ticker: `{ticker_symbol}` ---")
    
    try:
        stock = yf.Ticker(ticker_symbol)
        st.write(f"Debug: `yf.Ticker('{ticker_symbol}')` object created successfully.")

        # === Test 1: Attempt history() call FIRST ===
        st.write(f"Debug: TEST 1 - Attempting `stock.history(period='2d')` for `{ticker_symbol}`...")
        try:
            hist_data = stock.history(period="2d")
            if not hist_data.empty:
                st.success(f"Debug: TEST 1 - `stock.history(period='2d')` for `{ticker_symbol}` SUCCEEDED.")
                st.write(f"Debug: TEST 1 - Fetched {len(hist_data)} rows. Sample:")
                st.dataframe(hist_data.head(1))
            else:
                st.warning(f"Debug: TEST 1 - `stock.history(period='2d')` for `{ticker_symbol}` returned EMPTY DataFrame.")
        except Exception as e_hist_test:
            st.error(f"Debug: TEST 1 - `stock.history(period='2d')` for `{ticker_symbol}` FAILED with error: {e_hist_test}")
        st.write("--- End of TEST 1 (history) ---")

        # === Test 2: Attempt .info call ===
        st.write(f"Debug: TEST 2 - Attempting `stock.info` for `{ticker_symbol}`...")
        stock_info_data = None
        try:
            stock_info_data = stock.info
            if isinstance(stock_info_data, dict) and stock_info_data.get('symbol'):
                st.success(f"Debug: TEST 2 - Successfully retrieved `stock.info` for `{ticker_symbol}`.")
                # st.json(stock_info_data) # You can uncomment this if .info succeeds
                return stock_info_data # Success for .info
            else:
                st.warning(f"Debug: TEST 2 - `stock.info` for `{ticker_symbol}` was empty, invalid, or 'symbol' key missing.")
                st.json(stock_info_data if stock_info_data is not None else "`stock_info_data` is None (from .info test)")
                return None # .info was problematic, return None
        except Exception as e_info:
            st.error(f"Debug: TEST 2 - Error calling `stock.info` for `{ticker_symbol}`: {e_info}")
            return None # .info call failed, return None
            
    except Exception as e_ticker:
        st.error(f"Debug: Critical error creating `yf.Ticker('{ticker_symbol}')` object: {e_ticker}")
        return None


def get_stock_history(ticker_symbol, period="1y", interval="1d"):
    """Fetches historical market data using yfinance."""
    try:
        stock = yf.Ticker(ticker_symbol)
        history = stock.history(period=period, interval=interval)
        if history.empty:
            # Optional: log this
            # print(f"INFO: yfinance.Ticker('{ticker_symbol}').history returned empty DataFrame for period={period}, interval={interval}")
            return None
        return history
    except Exception as e:
        # Optional: log this error
        # print(f"ERROR: yfinance.Ticker('{ticker_symbol}').history call failed for period={period}, interval={interval}: {e}")
        return None

def get_financial_statements(ticker_symbol):
    """Fetches financial statements using yfinance."""
    try:
        stock = yf.Ticker(ticker_symbol)
        financials_data = {
            "annual_financials": stock.financials if hasattr(stock, 'financials') and not stock.financials.empty else pd.DataFrame(),
            "quarterly_financials": stock.quarterly_financials if hasattr(stock, 'quarterly_financials') and not stock.quarterly_financials.empty else pd.DataFrame(),
            "annual_balance_sheet": stock.balance_sheet if hasattr(stock, 'balance_sheet') and not stock.balance_sheet.empty else pd.DataFrame(),
            "quarterly_balance_sheet": stock.quarterly_balance_sheet if hasattr(stock, 'quarterly_balance_sheet') and not stock.quarterly_balance_sheet.empty else pd.DataFrame(),
            "annual_cashflow": stock.cashflow if hasattr(stock, 'cashflow') and not stock.cashflow.empty else pd.DataFrame(),
            "quarterly_cashflow": stock.quarterly_cashflow if hasattr(stock, 'quarterly_cashflow') and not stock.quarterly_cashflow.empty else pd.DataFrame(),
        }
        return financials_data
    except Exception as e:
        # Optional: log this error
        # print(f"ERROR: yfinance.Ticker('{ticker_symbol}') financial statements call failed: {e}")
        return { 
            "annual_financials": pd.DataFrame(), "quarterly_financials": pd.DataFrame(),
            "annual_balance_sheet": pd.DataFrame(), "quarterly_balance_sheet": pd.DataFrame(),
            "annual_cashflow": pd.DataFrame(), "quarterly_cashflow": pd.DataFrame(),
        }

def clean_value_from_nse_table(value_str):
    """Cleans numeric string from NSE FII/DII table cells."""
    if value_str is None: return 0.0
    text = str(value_str).strip().replace(',', '').replace('(', '-').replace(')', '')
    text = re.sub(r"[^0-9.-]", "", text) 
    if not text or text == "-" or text.lower() == 'na' or text.lower() == 'n.a.':
        return 0.0
    try:
        return float(text)
    except ValueError:
        # Optional: log this conversion issue
        # print(f"Could not convert '{value_str}' to float after cleaning. Cleaned text: '{text}'")
        return 0.0

def fetch_fii_dii_data_nse():
    """
    Attempts to fetch daily FII/DII provisional data from the NSE India reports page
    using Selenium. Returns a DataFrame with 'Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'.
    THIS SCRAPER IS FRAGILE AND DEPENDS ON NSE WEBSITE STRUCTURE.
    """
    if not SELENIUM_AVAILABLE:
        st.error("Selenium is not installed. Cannot fetch live FII/DII data from NSE.")
        return pd.DataFrame(columns=['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'])

    url = "https://www.nseindia.com/reports/fii-dii"
    driver = None
    default_return_df = pd.DataFrame(columns=['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'])

    try:
        # print("Selenium: Initializing WebDriver (trying Safari, then Chrome)...") # Removed
        try: 
            driver = webdriver.Safari()
            # print("Selenium: Safari WebDriver initialized.") # Removed
        except (WebDriverException, FileNotFoundError) as e_safari:
            # print(f"Selenium: Safari WebDriver failed: {e_safari}. Trying Chrome.") # Removed
            try: 
                chrome_options = ChromeOptions()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                                
                try:
                    driver = webdriver.Chrome(options=chrome_options) 
                except WebDriverException: 
                    raise 
                # print("Selenium: Chrome WebDriver initialized (headless).") # Removed
            except Exception as e_chrome:
                st.error(f"Selenium WebDriver Error: Could not start Safari or Chrome. Error: {e_chrome}")
                # print(f"CRITICAL Selenium: Both Safari and Chrome WebDriver failed: {e_chrome}") # Removed
                return default_return_df
        
        # print(f"Selenium: Fetching FII/DII data from: {url}") # Removed
        driver.get(url)
        
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "fiidiiTable")))
        # print("Selenium: Page loaded and table 'fiidiiTable' detected.") # Removed
        time.sleep(3) 

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        table = soup.find('table', id='fiidiiTable')

        fii_net, dii_net, report_date_from_table = None, None, None
        if table:
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')
                # print(f"Selenium: Found {len(rows)} rows in tbody.") # Removed
                if len(rows) >= 2: 
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) == 5: 
                            category_text = cols[0].text.strip().lower()
                            date_text = cols[1].text.strip()
                            net_value_text = cols[4].text.strip()
                            
                            if report_date_from_table is None and date_text:
                                try:
                                    report_date_from_table = pd.to_datetime(date_text, format='%d-%b-%Y', errors='raise').normalize()
                                except ValueError:
                                    # print(f"Warning: Could not parse date '{date_text}' from NSE table.") # Optional: log this
                                    pass
                            
                            current_net_val = clean_value_from_nse_table(net_value_text)
                            if "fii" in category_text or "fpi" in category_text:
                                fii_net = current_net_val
                                # print(f"Selenium: Extracted FII Net: {fii_net} for date text '{date_text}'") # Removed
                            elif "dii" in category_text:
                                dii_net = current_net_val
                                # print(f"Selenium: Extracted DII Net: {dii_net} for date text '{date_text}'") # Removed
                # else: # Removed
                    # print(f"Selenium: Expected at least 2 rows in table, found {len(rows)}.")
            # else: print("Selenium: tbody not found in fiidiiTable.") # Removed
        # else: print("Selenium: Table with id='fiidiiTable' NOT FOUND.") # Removed

        final_report_date = report_date_from_table
        if pd.isna(final_report_date):
            today = pd.Timestamp('today').normalize()
            final_report_date = today - pd.Timedelta(days=1 if today.dayofweek != 0 else 3) 
            # print(f"Selenium: Date not found in table, using fallback date: {final_report_date}") # Optional: log this

        if fii_net is not None and dii_net is not None:
            # print(f"Selenium: Successfully extracted FII_Net={fii_net}, DII_Net={dii_net} for date {final_report_date}") # Removed
            return pd.DataFrame({
                'Date': [final_report_date],
                'FII_Net_Purchase_Sales': [fii_net],
                'DII_Net_Purchase_Sales': [dii_net]
            })
        else:
            st.warning("Could not extract FII/DII net values from NSE page. Data might be outdated or scraper needs update.")
            # print("Selenium: Failed to extract both FII and DII net values reliably.") # Removed
            return default_return_df 

    except TimeoutException:
        st.error("Timeout waiting for NSE FII/DII data page to load elements. The website might be slow or changed.")
        # print("CRITICAL Selenium: TimeoutException on NSE FII/DII page.") # Removed
        return default_return_df
    except WebDriverException as wde:
        st.error(f"Selenium WebDriver error for NSE FII/DII: {wde}. Ensure WebDriver is correctly installed and in PATH.")
        # print(f"CRITICAL Selenium: WebDriverException for NSE FII/DII: {wde}") # Removed
        return default_return_df
    except Exception as e:
        st.error(f"An unexpected error occurred fetching NSE FII/DII data: {e}")
        # print(f"CRITICAL Selenium: Unexpected error in fetch_fii_dii_data_nse: {e}") # Removed
        return default_return_df
    finally:
        if driver:
            driver.quit()
            # print("Selenium: WebDriver quit.") # Removed

def fetch_nifty_pe_historical_data(url="https://www.finlive.in/page/nifty-50-nifty-pe-ratio/"):
    """ Fetches historical NIFTY 50 P/E data. THIS SCRAPER IS FRAGILE. """
    if not SELENIUM_AVAILABLE:
        st.error("Selenium is not installed. Cannot fetch live Nifty P/E data from Finlive.")
        return pd.DataFrame(columns=['Date', 'PE_Ratio'])
        
    # print(f"Attempting to fetch NIFTY P/E data using Selenium from: {url}") # Removed
    driver = None
    pe_data_list = []
    default_pe_df = pd.DataFrame(columns=['Date', 'PE_Ratio'])

    try: 
        # print("Selenium (P/E): Initializing WebDriver...") # Removed
        try:
            driver = webdriver.Safari()
            # print("Selenium (P/E): Safari WebDriver initialized.") # Removed
        except (WebDriverException, FileNotFoundError) as e_safari_pe:
            # print(f"Selenium (P/E): Safari WebDriver failed: {e_safari_pe}. Trying Chrome.") # Removed
            try:
                chrome_options_pe = ChromeOptions()
                chrome_options_pe.add_argument("--headless")
                chrome_options_pe.add_argument("--no-sandbox")
                chrome_options_pe.add_argument("--disable-dev-shm-usage")
                chrome_options_pe.add_argument("--disable-gpu")
                driver = webdriver.Chrome(options=chrome_options_pe) 
                # print("Selenium (P/E): Chrome WebDriver initialized (headless).") # Removed
            except Exception as e_chrome_pe:
                st.error(f"Selenium WebDriver Error (P/E): Could not start Safari or Chrome. Error: {e_chrome_pe}")
                # print(f"CRITICAL Selenium (P/E): Both Safari and Chrome WebDriver failed: {e_chrome_pe}") # Removed
                return default_pe_df
        
        driver.get(url)
        # print(f"Selenium (P/E): Page requested: {url}") # Removed
        target_table_locator = (By.CSS_SELECTOR, "div.table-responsive table.table") 
        WebDriverWait(driver, 45).until(EC.presence_of_element_located(target_table_locator))
        # print("Selenium (P/E): Table element found.") # Removed
        time.sleep(5)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        table_div = soup.find('div', class_='table-responsive')
        if not table_div:
            st.error(f"Error (P/E): Could not find 'div.table-responsive' on {url}. Website structure might have changed.")
            return default_pe_df
        data_table = table_div.find('table', class_='table')
        if not data_table:
            st.error(f"Error (P/E): Could not find table.table inside div.table-responsive on {url}.")
            return default_pe_df
        
        thead = data_table.find('thead')
        tbody = data_table.find('tbody')
        if not tbody or not thead:
            st.error(f"Error (P/E): Table header or body not found on {url}.")
            return default_pe_df

        header_row = thead.find('tr')
        if not header_row: st.error("Error (P/E): Header row not found."); return default_pe_df
            
        month_headers = [th.text.strip() for th in header_row.find_all('th')]
        if not month_headers or month_headers[0].lower() != 'year':
            st.error(f"Error (P/E): Table header format unexpected. Expected 'Year', got '{month_headers[0]}'.")
            return default_pe_df
        actual_month_names = month_headers[1:13] 

        for row in tbody.find_all('tr'):
            cells = row.find_all('td')
            if not cells or len(cells) == 0: continue
            year_str = cells[0].text.strip()
            try: year = int(year_str)
            except ValueError: continue

            for month_idx, month_name_from_header in enumerate(actual_month_names):
                cell_index = month_idx + 1
                if cell_index < len(cells):
                    pe_str = cells[cell_index].text.strip()
                    month_num = MONTH_TO_NUM.get(month_name_from_header)
                    if month_num and pe_str and pe_str not in ['-', '']:
                        try:
                            pe_value = float(pe_str)
                            current_date = datetime(year, month_num, 1) 
                            pe_data_list.append({'Date': current_date, 'PE_Ratio': pe_value})
                        except ValueError:
                            # Optional: log this
                            # print(f"Warning (P/E): Could not convert P/E value '{pe_str}' for {month_name_from_header} {year}.")
                            pass
                else: break 
        
        if not pe_data_list:
            st.warning(f"Warning (P/E): No P/E data extracted from table on {url}. Check website or data format.")
            return default_pe_df

        historical_pe_df = pd.DataFrame(pe_data_list)
        if not historical_pe_df.empty:
            historical_pe_df['Date'] = pd.to_datetime(historical_pe_df['Date'])
            historical_pe_df = historical_pe_df.sort_values(by="Date").drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        
        # print(f"Successfully fetched and parsed {len(historical_pe_df)} monthly P/E data points using Selenium from {url}.") # Removed
        return historical_pe_df

    except TimeoutException:
        st.error(f"Timeout (P/E): Waiting for P/E data table on {url}. Website might be slow or structure changed.")
        # print(f"CRITICAL Selenium (P/E): TimeoutException on {url}.") # Removed
        return default_pe_df
    except WebDriverException as wde_pe:
        st.error(f"Selenium WebDriver error for Nifty P/E: {wde_pe}.")
        # print(f"CRITICAL Selenium (P/E): WebDriverException: {wde_pe}") # Removed
        return default_pe_df
    except Exception as e:
        st.error(f"An unexpected error occurred fetching NIFTY P/E data: {e}")
        # print(f"CRITICAL Selenium (P/E): Unexpected error in fetch_nifty_pe_historical_data: {e}") # Removed
        # import traceback # Removed
        # traceback.print_exc() # Removed
        return default_pe_df
    finally:
        if driver:
            driver.quit()
            # print("Selenium (P/E): WebDriver quit.") # Removed