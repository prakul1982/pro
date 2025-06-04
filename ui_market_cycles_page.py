# ui_market_cycles_page.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import os
from gnews import GNews
# import re # Removed unused import

# --- Import your data fetching functions ---
from data_fetcher import (
    get_stock_history,
    fetch_fii_dii_data_nse,
    fetch_nifty_pe_historical_data
)

from utils import (
    load_csv_data,
    append_to_daily_fii_dii_csv, 
    save_csv_data
)

# --- Constants ---
PE_FILE_PATH = 'nifty_pe_ratio_live.csv'
FII_DATA_CSV_PATH = "historical_fii_data.csv" 
MARKET_CORRECTIONS_CSV_PATH = "market_corrections_data.csv"
FII_DII_DAILY_CSV_PATH = 'daily_fii_dii_nse_data.csv' 
NIFTY_TICKER = "^NSEI"

INDIA_VIX_CSV_PATH = "india_vix_data.csv"
PCR_CSV_PATH = "pcr_data.csv"
AD_LINE_CSV_PATH = "ad_line_data.csv"
NH_NL_CSV_PATH = "nh_nl_data.csv"
MARKET_CAP_GDP_CSV_PATH = "market_cap_gdp.csv"
GDP_GROWTH_CSV_PATH = "gdp_growth_data.csv"
CPI_INFLATION_CSV_PATH = "cpi_inflation_data.csv"
REPO_RATE_CSV_PATH = "repo_rate_data.csv"
IIP_DATA_CSV_PATH = "iip_data.csv"

# --- Configuration for P/E Ratio Bands ---
PE_BANDS = {
    "Very Undervalued": (0, 14),
    "Undervalued": (14, 18),
    "Fair Value": (18, 22),
    "Overvalued": (22, 26),
    "Very Overvalued": (26, float('inf'))
}
PE_BAND_COLORS = {
    "Very Undervalued": "rgba(0, 128, 0, 0.2)",
    "Undervalued": "rgba(144, 238, 144, 0.2)",
    "Fair Value": "rgba(255, 255, 0, 0.2)",
    "Overvalued": "rgba(255, 165, 0, 0.2)",
    "Very Overvalued": "rgba(255, 0, 0, 0.2)"
}
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# --- Default Column Lists for load_csv_data ---
PE_DEFAULT_COLS = ['Date', 'PE_Ratio']
FII_HISTORICAL_DEFAULT_COLS = ['Date', 'FII_Total_Net_Investment']
FII_DII_DAILY_DEFAULT_COLS = ['Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales']
CORRECTIONS_DEFAULT_COLS = ["Event_Name", "Approx_Peak_Month_Str", "Approx_Trough_Month_Str",
                            "Leading_Indicators_Signals", "FII_Behavior_Qualitative", "Key_Reasons"]

# --- Data Loading Functions ---
@st.cache_data(ttl=3*60*60)
def get_cached_latest_fpi_data():
    """
    Wrapper to cache the fetched daily FPI data from NSE.
    This function MUST ensure the output DataFrame has exactly:
    'Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'.
    """
    df_raw = fetch_fii_dii_data_nse() 

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=FII_DII_DAILY_DEFAULT_COLS)

    df_standardized = pd.DataFrame()

    if 'Date' in df_raw.columns:
        df_standardized['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
    else:
        return pd.DataFrame(columns=FII_DII_DAILY_DEFAULT_COLS)

    if 'FII_Net_Purchase_Sales' in df_raw.columns:
        df_standardized['FII_Net_Purchase_Sales'] = pd.to_numeric(
            df_raw['FII_Net_Purchase_Sales'].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
    elif 'FII Net Purchase Or Sale' in df_raw.columns: 
         df_standardized['FII_Net_Purchase_Sales'] = pd.to_numeric(
            df_raw['FII Net Purchase Or Sale'].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
    else:
        df_standardized['FII_Net_Purchase_Sales'] = pd.NA

    if 'DII_Net_Purchase_Sales' in df_raw.columns:
        df_standardized['DII_Net_Purchase_Sales'] = pd.to_numeric(
            df_raw['DII_Net_Purchase_Sales'].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
    elif 'DII Net Purchase Or Sale' in df_raw.columns: 
        df_standardized['DII_Net_Purchase_Sales'] = pd.to_numeric(
            df_raw['DII Net Purchase Or Sale'].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
    else:
        df_standardized['DII_Net_Purchase_Sales'] = pd.NA

    df_standardized.dropna(subset=['Date'], inplace=True) 
    
    final_df = pd.DataFrame(columns=FII_DII_DAILY_DEFAULT_COLS)
    for col in FII_DII_DAILY_DEFAULT_COLS:
        if col in df_standardized.columns:
            final_df[col] = df_standardized[col]
        elif col == 'Date': 
            final_df[col] = pd.NaT
        else:
            final_df[col] = pd.NA 
            
    return final_df

def load_pe_data():
    df = load_csv_data(PE_FILE_PATH, default_cols=PE_DEFAULT_COLS, parse_dates_cols=['Date'], error_level="warning")
    if not df.empty:
        df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def load_fii_dii_historical_data():
    df = load_csv_data(FII_DATA_CSV_PATH, default_cols=FII_HISTORICAL_DEFAULT_COLS, parse_dates_cols=['Date'], error_level="warning", thousands=',')
    if not df.empty:
        if 'FII_Total_Net_Investment' in df.columns:
            df['FII_Total_Net_Investment'] = pd.to_numeric(df['FII_Total_Net_Investment'], errors='coerce')
        df = df.sort_values(by="Date").reset_index(drop=True)
    return df

def load_daily_fii_dii_data():
    """
    Loads daily FII/DII data from FII_DII_DAILY_CSV_PATH.
    Focuses on 'Date', 'FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales'.
    """
    df = load_csv_data(
        FII_DII_DAILY_CSV_PATH,
        default_cols=FII_DII_DAILY_DEFAULT_COLS, 
        parse_dates_cols=['Date'],
        error_level="warning",
        thousands=',' 
    )
    
    if not df.empty:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        
        for col_name in ['FII_Net_Purchase_Sales', 'DII_Net_Purchase_Sales']:
            if col_name in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name].astype(str).str.replace(',', '', regex=False)
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                df[col_name] = pd.NA 
        
        df = df.sort_values(by='Date').reset_index(drop=True)
        return df.reindex(columns=FII_DII_DAILY_DEFAULT_COLS)
        
    return pd.DataFrame(columns=FII_DII_DAILY_DEFAULT_COLS)

def load_market_correction_data():
    potential_date_cols = ['Start Date', 'End Date', 'Peak Date', 'Bottom Date']
    actual_cols_in_file = []
    if os.path.exists(MARKET_CORRECTIONS_CSV_PATH):
        try:
            temp_df = pd.read_csv(MARKET_CORRECTIONS_CSV_PATH, nrows=0)
            actual_cols_in_file = temp_df.columns.tolist()
        except Exception:
            pass
            
    date_columns_to_parse = [col for col in potential_date_cols if col in actual_cols_in_file]
    
    df = load_csv_data(
        MARKET_CORRECTIONS_CSV_PATH,
        default_cols=CORRECTIONS_DEFAULT_COLS,
        parse_dates_cols=date_columns_to_parse,
        error_level="warning"
    )
    return df

# --- Placeholder for Market Cycle Logic ---
def determine_market_phase(pe_data, fii_data, vix_data, pcr_data, ad_data, nhnl_data, sectoral_pes, market_cap_gdp, gdp_growth, inflation, interest_rates, iip_data, banking_health_data):
    if pe_data is not None and not pe_data.empty and 'PE_Ratio' in pe_data.columns:
        current_pe = pe_data['PE_Ratio'].iloc[-1]
        if current_pe > 25: return "Overheated/Peak"
        if current_pe < 15: return "Undervalued/Trough"
        return "Fair Value/Expansion"
    return "Data Pending"

def plot_market_cycle_clock(phase):
    st.metric("Estimated Market Phase", phase)


# --- Main Page Function ---
def display_market_cycles_page():
    st.title("📈 Market Cycles Analysis")
    st.markdown("""
    This dashboard aims to provide insights into the current phase of the Indian stock market 
    by analyzing various valuation, sentiment, economic, and flow indicators.
    """)
    st.markdown("---")

    nifty_pe_df = load_pe_data()
    daily_fpi_df_loaded = load_daily_fii_dii_data()
    combined_fii_df = load_fii_dii_historical_data()
    corrections_df_loaded = load_market_correction_data()
    
    vix_df = load_csv_data(INDIA_VIX_CSV_PATH, default_cols=['Date', 'VIX_Close'], parse_dates_cols=['Date'], error_level="info")
    pcr_df = load_csv_data(PCR_CSV_PATH, default_cols=['Date', 'PCR'], parse_dates_cols=['Date'], error_level="info")
    ad_line_df = load_csv_data(AD_LINE_CSV_PATH, default_cols=['Date', 'AD_Line'], parse_dates_cols=['Date'], error_level="info")
    nh_nl_df = load_csv_data(NH_NL_CSV_PATH, default_cols=['Date', 'New_Highs', 'New_Lows'], parse_dates_cols=['Date'], error_level="info")
    market_cap_gdp_df = load_csv_data(MARKET_CAP_GDP_CSV_PATH, default_cols=['Date', 'MarketCapGDP'], parse_dates_cols=['Date'], error_level="info")
    gdp_df = load_csv_data(GDP_GROWTH_CSV_PATH, default_cols=['Date', 'GDP_Growth'], parse_dates_cols=['Date'], error_level="info")
    cpi_df = load_csv_data(CPI_INFLATION_CSV_PATH, default_cols=['Date', 'CPI_Inflation'], parse_dates_cols=['Date'], error_level="info")
    repo_df = load_csv_data(REPO_RATE_CSV_PATH, default_cols=['Date', 'Repo_Rate'], parse_dates_cols=['Date'], error_level="info")
    iip_df = load_csv_data(IIP_DATA_CSV_PATH, default_cols=['Date', 'IIP_Growth'], parse_dates_cols=['Date'], error_level="info")
    
    st.header("🚦 Current Market Cycle Phase")
    current_phase = determine_market_phase(
        nifty_pe_df, combined_fii_df, vix_df, pcr_df, ad_line_df, nh_nl_df,
        {}, market_cap_gdp_df, gdp_df, cpi_df, repo_df, iip_df, pd.DataFrame()
    )
    plot_market_cycle_clock(current_phase)
    st.info(f"Based on the available data, the market is estimated to be in the **{current_phase}** phase. Detailed indicators below.")
    st.markdown("---")

    with st.expander("Valuation: NIFTY 50 P/E Ratio", expanded=True):
        st.markdown(f"Data from Finlive.in, stored in `{PE_FILE_PATH}`.")
        if st.button("🔄 Refresh NIFTY 50 P/E Data", key="refresh_nifty_pe_finlive_mc_v3"):
            with st.spinner("Fetching NIFTY P/E data from Finlive.in..."):
                pe_data_df_new = fetch_nifty_pe_historical_data()
                if pe_data_df_new is not None and not pe_data_df_new.empty:
                    save_csv_data(pe_data_df_new, PE_FILE_PATH, date_cols_to_format=['Date'])
                    st.rerun()
                else:
                    st.error("Failed to fetch NIFTY P/E data from Finlive.in.")

        if nifty_pe_df.empty:
            st.warning(f"NIFTY 50 P/E data file ('{PE_FILE_PATH}') not found or is empty. Please use the 'Refresh' button or manage in Data Management Hub.")
        else:
            st.subheader("Historical NIFTY 50 P/E Ratio Chart")
            fig_pe_chart = go.Figure()
            fig_pe_chart.add_trace(go.Scatter(x=nifty_pe_df['Date'], y=nifty_pe_df['PE_Ratio'], mode='lines', name='NIFTY 50 P/E', line=dict(color='blue')))
            max_pe_value_chart = nifty_pe_df['PE_Ratio'].max() if not nifty_pe_df['PE_Ratio'].empty else 50
            for band_name, (lower_bound, upper_bound) in PE_BANDS.items():
                plot_upper_bound = upper_bound if upper_bound != float('inf') else max_pe_value_chart + 10
                fig_pe_chart.add_hrect(y0=lower_bound, y1=plot_upper_bound, fillcolor=PE_BAND_COLORS[band_name], opacity=0.3, layer="below", line_width=0, annotation_text=band_name, annotation_position="left", annotation_font_size=10, annotation_font_color="black")
            
            if not nifty_pe_df.empty:
                current_pe_record = nifty_pe_df.iloc[-1]
                current_nifty_pe_value = current_pe_record['PE_Ratio']
                current_nifty_pe_date = current_pe_record['Date']
                st.metric(label=f"Latest NIFTY 50 P/E ({current_nifty_pe_date.strftime('%b %Y')})", value=f"{current_nifty_pe_value:.2f}")
                fig_pe_chart.add_annotation(x=current_nifty_pe_date, y=current_nifty_pe_value, text=f"Latest P/E: {current_nifty_pe_value:.2f}<br>({current_nifty_pe_date.strftime('%b %Y')})", showarrow=True, arrowhead=1, ax=20, ay=-40)
                temp_pe_for_mean = nifty_pe_df.set_index('Date')['PE_Ratio'].copy()
                mean_pe_5y = temp_pe_for_mean.rolling(window=5*12, min_periods=int(3*12)).mean().iloc[-1] if len(temp_pe_for_mean) >= int(3*12) else None
                mean_pe_10y = temp_pe_for_mean.rolling(window=10*12, min_periods=int(7*12)).mean().iloc[-1] if len(temp_pe_for_mean) >= int(7*12) else None
                col1_pe_avg, col2_pe_avg = st.columns(2)
                with col1_pe_avg: st.metric(label="Approx. 5-Year Avg P/E", value=f"{mean_pe_5y:.2f}" if mean_pe_5y else "N/A")
                with col2_pe_avg: st.metric(label="Approx. 10-Year Avg P/E", value=f"{mean_pe_10y:.2f}" if mean_pe_10y else "N/A")
                if mean_pe_10y and current_nifty_pe_value is not None:
                    deviation = ((current_nifty_pe_value - mean_pe_10y) / mean_pe_10y) * 100
                    st.write(f"Latest P/E is **{deviation:.2f}%** {'above' if deviation > 0 else 'below'} the 10-year monthly average.")
                if mean_pe_5y: fig_pe_chart.add_hline(y=mean_pe_5y, line_dash="dash", line_color="orange", annotation_text=f"5Y Avg: {mean_pe_5y:.2f}", annotation_position="bottom right")
                if mean_pe_10y: fig_pe_chart.add_hline(y=mean_pe_10y, line_dash="dash", line_color="green", annotation_text=f"10Y Avg: {mean_pe_10y:.2f}", annotation_position="top right")
            fig_pe_chart.update_layout(title='NIFTY 50 P/E Ratio Over Time with Valuation Bands', xaxis_title='Date', yaxis_title='P/E Ratio', hovermode="x unified", height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pe_chart, use_container_width=True)

            st.subheader("NIFTY 50 P/E Ratio Monthly Heatmap")
            pe_data_for_heatmap = nifty_pe_df.copy()
            pe_data_for_heatmap['Year'] = pe_data_for_heatmap['Date'].dt.year
            pe_data_for_heatmap['Month'] = pe_data_for_heatmap['Date'].dt.strftime('%b')
            try:
                heatmap_data = pe_data_for_heatmap.pivot_table(index='Year', columns='Month', values='PE_Ratio')
                heatmap_data = heatmap_data.reindex(columns=MONTH_ORDER)
                heatmap_data = heatmap_data.sort_index(ascending=False)
                if not heatmap_data.empty:
                    fig_heatmap = px.imshow(heatmap_data, text_auto='.2f', aspect="auto", color_continuous_scale=px.colors.diverging.RdYlGn_r, labels=dict(x="Month", y="Year", color="P/E Ratio"), height=max(400, len(heatmap_data.index) * 35 + 60))
                    fig_heatmap.update_xaxes(side="top", tickangle=0)
                    fig_heatmap.update_layout(title_text='NIFTY 50 Monthly P/E Ratio Heatmap', title_x=0.5, xaxis_title=None, yaxis_title=None)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else: st.write("Not enough data to display P/E heatmap (after pivoting).")
            except Exception as e: st.error(f"Error creating P/E heatmap: {e}")
    st.markdown("---")

    with st.expander("Investor Flows: FII/DII Activity", expanded=False):
        st.markdown("""Analyzes daily provisional data (from NSE) and historical monthly FII investment data.""")
        
        st.subheader("Daily FII/DII Net Investment")
        st.markdown(f"Data stored in `{FII_DII_DAILY_CSV_PATH}`. Expected columns: `Date`, `FII_Net_Purchase_Sales`, `DII_Net_Purchase_Sales`.")
        
        if st.button("🔄 Refresh Daily FII/DII Data", key="fetch_daily_fpi_mc_v4"):
            with st.spinner("Fetching latest daily FII/DII data..."):
                new_daily_data = get_cached_latest_fpi_data() 
                if new_daily_data is not None and not new_daily_data.empty:
                    if append_to_daily_fii_dii_csv(new_daily_data, file_path=FII_DII_DAILY_CSV_PATH):
                        st.toast(f"Daily FII/DII data updated in {FII_DII_DAILY_CSV_PATH}.", icon="💾")
                    else:
                        st.error(f"Failed to update daily FII/DII data CSV ({FII_DII_DAILY_CSV_PATH}). Check utils.py and scraper output.")
                    st.rerun()
                else:
                    st.warning("Could not fetch fresh daily FII/DII data. Scraper might need attention or returned no data.")

        if daily_fpi_df_loaded is not None and not daily_fpi_df_loaded.empty:
            st.caption("Latest available daily figures (showing last 5 days):")
            display_cols = [col for col in FII_DII_DAILY_DEFAULT_COLS if col in daily_fpi_df_loaded.columns]
            display_df_daily = daily_fpi_df_loaded[display_cols].copy()
            
            if 'Date' in display_df_daily.columns:
                try:
                    display_df_daily['Date'] = pd.to_datetime(display_df_daily['Date'], errors='coerce').dt.strftime('%d-%b-%Y')
                except AttributeError:
                    pass 
            st.dataframe(display_df_daily.tail())

            plot_df_daily = daily_fpi_df_loaded.copy()
            if 'Date' in plot_df_daily.columns:
                 plot_df_daily['Date'] = pd.to_datetime(plot_df_daily['Date'], errors='coerce')
                 plot_df_daily.dropna(subset=['Date'], inplace=True)

            if 'FII_Net_Purchase_Sales' in plot_df_daily.columns:
                 plot_df_daily['FII_Net_Purchase_Sales'] = pd.to_numeric(plot_df_daily['FII_Net_Purchase_Sales'], errors='coerce')
            if 'DII_Net_Purchase_Sales' in plot_df_daily.columns:
                 plot_df_daily['DII_Net_Purchase_Sales'] = pd.to_numeric(plot_df_daily['DII_Net_Purchase_Sales'], errors='coerce')
            
            fig_daily_net = go.Figure()
            traces_added_daily = False
            
            fii_net_col = 'FII_Net_Purchase_Sales'
            dii_net_col = 'DII_Net_Purchase_Sales'

            if fii_net_col in plot_df_daily.columns and plot_df_daily[fii_net_col].notna().any():
                fig_daily_net.add_trace(go.Bar(x=plot_df_daily['Date'], y=plot_df_daily[fii_net_col], name='FPI Net'))
                traces_added_daily = True
            
            if dii_net_col in plot_df_daily.columns and plot_df_daily[dii_net_col].notna().any():
                fig_daily_net.add_trace(go.Bar(x=plot_df_daily['Date'], y=plot_df_daily[dii_net_col], name='DII Net'))
                traces_added_daily = True

            if traces_added_daily:
                fig_daily_net.update_layout(title="Daily Net Investments (Provisional)", barmode='group', xaxis_title="Date", yaxis_title="Net Investment (INR Cr)")
                st.plotly_chart(fig_daily_net, use_container_width=True)
            else:
                st.info(f"No valid data to plot for FII/DII net flows. Check if '{fii_net_col}' and/or '{dii_net_col}' columns exist and contain numeric data in `{FII_DII_DAILY_CSV_PATH}`.")
        else:
            st.info(f"Daily FII/DII data ('{FII_DII_DAILY_CSV_PATH}') is empty or not found. Refresh or check Data Management Hub.")

        st.subheader("Monthly FII Net Investment Trends & Market Correlation")
        st.markdown(f"Data from `{FII_DATA_CSV_PATH}`.")
        if not combined_fii_df.empty and 'FII_Total_Net_Investment' in combined_fii_df.columns:
            display_df_fii_recent = combined_fii_df[['Date', 'FII_Total_Net_Investment']].copy()
            display_df_fii_recent['Date_Display'] = display_df_fii_recent['Date'].dt.strftime('%B %Y')
            st.caption("Recent Monthly FII Net Investments:")
            st.dataframe(display_df_fii_recent[['Date_Display', 'FII_Total_Net_Investment']].tail().rename(columns={'Date_Display': 'Month-Year', 'FII_Total_Net_Investment': 'Total Net (INR Cr)'}).set_index('Month-Year'))

            merged_df_for_analysis = None
            with st.spinner(f"Fetching NIFTY 50 data for FII correlation analysis..."):
                nifty_hist_daily_fii = get_stock_history(NIFTY_TICKER, period="max", interval="1d")

            if nifty_hist_daily_fii is not None and not nifty_hist_daily_fii.empty:
                nifty_hist_daily_fii.index = pd.to_datetime(nifty_hist_daily_fii.index).tz_localize(None)
                nifty_monthly_close = nifty_hist_daily_fii['Close'].resample('MS').last().ffill()
                nifty_monthly_df = nifty_monthly_close.to_frame(name='Nifty_Close').reset_index()
                nifty_monthly_df['Date'] = pd.to_datetime(nifty_monthly_df['Date']).dt.normalize()

                fii_data_for_merge = combined_fii_df.copy()
                fii_data_for_merge['Date'] = pd.to_datetime(fii_data_for_merge['Date']).dt.to_period('M').dt.to_timestamp('s').dt.normalize()
                merged_df_for_analysis = pd.merge(fii_data_for_merge, nifty_monthly_df, on='Date', how='left')

                if not merged_df_for_analysis.empty and 'FII_Total_Net_Investment' in merged_df_for_analysis.columns and 'Nifty_Close' in merged_df_for_analysis.columns:
                    merged_df_for_analysis['Cumulative_FII_Investment'] = merged_df_for_analysis['FII_Total_Net_Investment'].cumsum()
                    
                    st.write("**Monthly FII Net Investment vs. NIFTY 50**")
                    fig_overlay = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_overlay.add_trace(go.Bar(x=merged_df_for_analysis['Date'], y=merged_df_for_analysis['FII_Total_Net_Investment'], name='Monthly FII Net', marker_color='rgba(0,100,200,0.6)'), secondary_y=False,)
                    fig_overlay.add_trace(go.Scatter(x=merged_df_for_analysis['Date'], y=merged_df_for_analysis['Nifty_Close'], name='NIFTY 50', line=dict(color='orange')), secondary_y=True,)
                    fig_overlay.update_layout(hovermode="x unified")
                    fig_overlay.update_yaxes(title_text="FII Net Investment (INR Cr)", secondary_y=False)
                    fig_overlay.update_yaxes(title_text="NIFTY 50 Index", secondary_y=True)
                    st.plotly_chart(fig_overlay, use_container_width=True)

                    st.write("**Cumulative FII Net Investment vs. NIFTY 50**")
                    fig_cumulative = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_cumulative.add_trace(go.Scatter(x=merged_df_for_analysis['Date'], y=merged_df_for_analysis['Cumulative_FII_Investment'], name='Cumulative FII Net', line=dict(color='green')), secondary_y=False,)
                    fig_cumulative.add_trace(go.Scatter(x=merged_df_for_analysis['Date'], y=merged_df_for_analysis['Nifty_Close'], name='NIFTY 50', line=dict(color='orange')), secondary_y=True,)
                    fig_cumulative.update_layout(hovermode="x unified")
                    fig_cumulative.update_yaxes(title_text="Cumulative FII Net (INR Cr)", secondary_y=False)
                    fig_cumulative.update_yaxes(title_text="NIFTY 50 Index", secondary_y=True)
                    st.plotly_chart(fig_cumulative, use_container_width=True)

                    st.write("**Rolling Correlation: FII Net Flow vs. NIFTY 50 Monthly Returns**")
                    rolling_window_corr = st.slider("Select Rolling Window (Months):", min_value=3, max_value=24, value=6, step=1, key="fii_corr_window_mc_final_v9") 
                    
                    merged_df_for_analysis['Nifty_Monthly_Return'] = merged_df_for_analysis['Nifty_Close'].pct_change()
                    corr_df = merged_df_for_analysis[['Date', 'FII_Total_Net_Investment', 'Nifty_Monthly_Return']].copy() 
                    corr_df.set_index('Date', inplace=True)
                    corr_df.dropna(inplace=True)

                    if len(corr_df) >= rolling_window_corr :
                        rolling_corr_values = corr_df['FII_Total_Net_Investment'].rolling(window=rolling_window_corr).corr(corr_df['Nifty_Monthly_Return'])
                        fig_rolling_corr = px.line(x=rolling_corr_values.index, y=rolling_corr_values, title=f'{rolling_window_corr}-Month Rolling Correlation')
                        fig_rolling_corr.add_hline(y=0, line_dash="dash", line_color="grey")
                        fig_rolling_corr.update_yaxes(title_text="Correlation Coefficient", range=[-1, 1])
                        st.plotly_chart(fig_rolling_corr, use_container_width=True)
                    else: 
                        st.info(f"Not enough data points ({len(corr_df)}) for {rolling_window_corr}-month rolling correlation.")
                else:
                    st.warning("Could not merge FII data with Nifty data for correlation analysis. Check date alignment and data availability.")
            else:
                st.warning("Could not fetch NIFTY 50 historical data for FII analysis.")
        else:
            st.info(f"Monthly FII data not available in '{FII_DATA_CSV_PATH}'. Please manage it in the Data Management Hub.")
    st.markdown("---")

    with st.expander("Market Sentiment & Breadth Indicators", expanded=False):
        st.markdown("Gauging market mood and participation. Data for these indicators needs to be fetched or uploaded via the Data Management Hub.")
        
        col_sent1, col_sent2 = st.columns(2)
        with col_sent1:
            st.subheader("India VIX")
            if not vix_df.empty:
                fig_vix = px.line(vix_df, x='Date', y='VIX_Close', title='India VIX Trend')
                st.plotly_chart(fig_vix, use_container_width=True)
            else:
                st.info(f"India VIX data ({INDIA_VIX_CSV_PATH}) not available. Create/upload the file or implement the scraper.")

            st.subheader("Advance-Decline Line")
            if not ad_line_df.empty:
                fig_ad = px.line(ad_line_df, x='Date', y='AD_Line', title='Advance-Decline Line')
                st.plotly_chart(fig_ad, use_container_width=True)
            else:
                st.info(f"Advance-Decline Line data ({AD_LINE_CSV_PATH}) not available. Create/upload the file or implement the scraper.")

        with col_sent2:
            st.subheader("Put-Call Ratio (PCR)")
            if not pcr_df.empty:
                fig_pcr = px.line(pcr_df, x='Date', y='PCR', title='NIFTY Put-Call Ratio')
                st.plotly_chart(fig_pcr, use_container_width=True)
            else:
                st.info(f"PCR data ({PCR_CSV_PATH}) not available. Create/upload the file or implement the scraper.")

            st.subheader("New Highs vs. New Lows")
            if not nh_nl_df.empty:
                fig_nhnl = go.Figure()
                if 'New_Highs' in nh_nl_df.columns:
                    fig_nhnl.add_trace(go.Bar(x=nh_nl_df['Date'], y=nh_nl_df['New_Highs'], name='New Highs'))
                if 'New_Lows' in nh_nl_df.columns:
                    fig_nhnl.add_trace(go.Bar(x=nh_nl_df['Date'], y=nh_nl_df['New_Lows'], name='New Lows'))
                fig_nhnl.update_layout(title="New Highs vs. New Lows", barmode='group')
                st.plotly_chart(fig_nhnl, use_container_width=True)
            else:
                st.info(f"New Highs-New Lows data ({NH_NL_CSV_PATH}) not available. Create/upload the file or implement the scraper.")
    st.markdown("---")
    
    with st.expander("Sectoral Analysis", expanded=False):
        st.markdown("P/E Ratios and Performance of key NIFTY Sectors. This section requires data for individual sectors (e.g., `sectoral_pe_bank.csv`, `sectoral_performance.csv`).")
        st.info("Detailed sectoral P/E and performance analysis coming soon. Data needs to be sourced and scrapers built.")
    st.markdown("---")

    with st.expander("Economic Health Indicators", expanded=False):
        st.markdown("Key macroeconomic indicators. Data for these indicators needs to be sourced/updated from their respective CSV files.")
        
        col_econ1, col_econ2 = st.columns(2)
        with col_econ1:
            st.subheader("Market Cap to GDP Ratio")
            if not market_cap_gdp_df.empty:
                fig_mcap_gdp = px.line(market_cap_gdp_df, x='Date', y='MarketCapGDP', title='Market Cap to GDP Ratio')
                st.plotly_chart(fig_mcap_gdp, use_container_width=True)
            else:
                st.info(f"Market Cap to GDP data ('{MARKET_CAP_GDP_CSV_PATH}') not available.")

            st.subheader("CPI Inflation Rate (YoY %)")
            if not cpi_df.empty:
                fig_cpi = px.line(cpi_df, x='Date', y='CPI_Inflation', title='CPI Inflation Rate (YoY)')
                st.plotly_chart(fig_cpi, use_container_width=True)
            else:
                st.info(f"CPI Inflation data ('{CPI_INFLATION_CSV_PATH}') not available.")

        with col_econ2:
            st.subheader("GDP Growth Rate (YoY %)")
            if not gdp_df.empty:
                fig_gdp = px.line(gdp_df, x='Date', y='GDP_Growth', title='GDP Growth Rate (YoY)')
                st.plotly_chart(fig_gdp, use_container_width=True)
            else:
                st.info(f"GDP Growth data ('{GDP_GROWTH_CSV_PATH}') not available.")

            st.subheader("Interest Rates (RBI Repo Rate)")
            if not repo_df.empty:
                fig_repo = px.line(repo_df, x='Date', y='Repo_Rate', title='RBI Repo Rate')
                st.plotly_chart(fig_repo, use_container_width=True)
            else:
                st.info(f"Repo Rate data ('{REPO_RATE_CSV_PATH}') not available.")
        
        st.subheader("Index of Industrial Production (IIP)")
        if not iip_df.empty:
            fig_iip = px.line(iip_df, x='Date', y='IIP_Growth', title='IIP Growth Rate (YoY)')
            st.plotly_chart(fig_iip, use_container_width=True)
        else:
            st.info(f"IIP data ('{IIP_DATA_CSV_PATH}') not available.")
    st.markdown("---")

    with st.expander("Banking Sector Health & Sentiment", expanded=False):
        st.info("Analysis of Nifty Bank performance, P/E, and key banking metrics will be developed here. Data needs to be sourced (e.g., `banking_sector_data.csv`).")
    st.markdown("---")

    with st.expander("Review of Past Market Corrections", expanded=False):
        st.subheader("Notable Past Market Corrections (Indian Context)")
        
        nifty_history_for_corrections_full_corr = get_stock_history(NIFTY_TICKER, period="max", interval="1d")
        nifty_history_for_corrections = None
        if nifty_history_for_corrections_full_corr is not None and not nifty_history_for_corrections_full_corr.empty:
            nifty_history_for_corrections = nifty_history_for_corrections_full_corr.copy()
            nifty_history_for_corrections.index = pd.to_datetime(nifty_history_for_corrections.index).tz_localize(None)

        if not corrections_df_loaded.empty:
            for index, correction_row in corrections_df_loaded.iterrows():
                event_name = str(correction_row.get("Event_Name", f"Unnamed Event {index + 1}"))
                st.markdown(f"#### {event_name}")

                peak_month_str = correction_row.get("Approx_Peak_Month_Str")
                trough_month_str = correction_row.get("Approx_Trough_Month_Str")
                peak_month_dt = pd.to_datetime(peak_month_str, format='%Y-%m', errors='coerce') if pd.notna(peak_month_str) else None
                trough_month_dt = pd.to_datetime(trough_month_str, format='%Y-%m', errors='coerce') if pd.notna(trough_month_str) else None
                nifty_peak_val, nifty_trough_val, actual_peak_date, actual_trough_date, nifty_drop_percent, fii_flow_during_correction = None, None, None, None, None, None

                if nifty_history_for_corrections is not None and not nifty_history_for_corrections.empty and pd.notna(peak_month_dt) and pd.notna(trough_month_dt):
                    analysis_window_start = peak_month_dt - pd.DateOffset(months=3)
                    analysis_window_end = trough_month_dt + pd.DateOffset(months=3) + pd.offsets.MonthEnd(0)
                    nifty_period_data = nifty_history_for_corrections[(nifty_history_for_corrections.index >= analysis_window_start) & (nifty_history_for_corrections.index <= analysis_window_end)]
                    if not nifty_period_data.empty:
                        peak_search_end_date = trough_month_dt + pd.offsets.MonthEnd(0)
                        peak_search_data = nifty_period_data[nifty_period_data.index <= peak_search_end_date]
                        if not peak_search_data.empty and not peak_search_data['Close'].dropna().empty:
                            actual_peak_date_ts = peak_search_data['Close'].idxmax()
                            nifty_peak_val = peak_search_data.loc[actual_peak_date_ts, 'Close']
                            actual_peak_date = pd.to_datetime(actual_peak_date_ts).date()
                            if actual_peak_date:
                                trough_search_data = nifty_period_data[(nifty_period_data.index >= actual_peak_date_ts) & (nifty_period_data.index <= analysis_window_end)]
                                if not trough_search_data.empty and not trough_search_data['Close'].dropna().empty:
                                    actual_trough_date_ts = trough_search_data['Close'].idxmin()
                                    nifty_trough_val = trough_search_data.loc[actual_trough_date_ts, 'Close']
                                    actual_trough_date = pd.to_datetime(actual_trough_date_ts).date()
                                    if nifty_peak_val and nifty_peak_val != 0:
                                        nifty_drop_percent = ((nifty_trough_val - nifty_peak_val) / nifty_peak_val) * 100
                
                if actual_peak_date and actual_trough_date and combined_fii_df is not None and not combined_fii_df.empty and 'FII_Total_Net_Investment' in combined_fii_df.columns:
                    fii_start_compare_date = pd.Timestamp(actual_peak_date).replace(day=1)
                    fii_end_compare_date = (pd.Timestamp(actual_trough_date).replace(day=1) + pd.offsets.MonthEnd(0))
                    fii_during_correction_df = combined_fii_df[(combined_fii_df['Date'] >= fii_start_compare_date) & (combined_fii_df['Date'] <= fii_end_compare_date)]
                    if not fii_during_correction_df.empty:
                        fii_flow_during_correction = fii_during_correction_df['FII_Total_Net_Investment'].sum()

                col1_details, col2_details = st.columns(2)
                with col1_details:
                    st.markdown(f"**Approx. Period:** {correction_row.get('Approx_Peak_Month_Str','N/A')} to {correction_row.get('Approx_Trough_Month_Str','N/A')}")
                    if actual_peak_date and actual_trough_date: st.markdown(f"**Identified Fall:** {actual_peak_date.strftime('%b %d, %Y')} to {actual_trough_date.strftime('%b %d, %Y')}")
                    peak_str = f"{nifty_peak_val:,.0f}" if nifty_peak_val is not None else "N/A"
                    trough_str = f"{nifty_trough_val:,.0f}" if nifty_trough_val is not None else "N/A"
                    if nifty_drop_percent is not None: st.markdown(f"**Nifty Drop:** {nifty_drop_percent:.2f}% (Peak: {peak_str}, Trough: {trough_str})")
                    if fii_flow_during_correction is not None: st.markdown(f"**Net FII Flow During Fall:** {fii_flow_during_correction:,.2f} Cr")
                    elif correction_row.get("FII_Behavior_Qualitative"): st.markdown(f"**FII Behavior (General):** {correction_row['FII_Behavior_Qualitative']}")
                with col2_details:
                    st.markdown(f"**Leading Indicators/Signals:** {correction_row.get('Leading_Indicators_Signals', 'N/A')}")
                    st.markdown(f"**Key Reasons:** {correction_row.get('Key_Reasons', 'N/A')}")

                if nifty_history_for_corrections is not None and actual_peak_date and actual_trough_date and nifty_peak_val is not None and nifty_trough_val is not None:
                    chart_plot_start = pd.Timestamp(actual_peak_date) - pd.DateOffset(days=60)
                    chart_plot_end = pd.Timestamp(actual_trough_date) + pd.DateOffset(days=60)
                    nifty_chart_data = nifty_history_for_corrections[
                        (nifty_history_for_corrections.index >= chart_plot_start) &
                        (nifty_history_for_corrections.index <= chart_plot_end)
                    ]['Close']
                    if not nifty_chart_data.empty:
                        st.markdown(f"##### NIFTY 50 during {event_name}")
                        fig_corr_nifty = px.line(x=nifty_chart_data.index, y=nifty_chart_data)
                        fig_corr_nifty.update_yaxes(title_text="NIFTY 50 Index")
                        fig_corr_nifty.add_trace(go.Scatter(x=[actual_peak_date, actual_trough_date],
                                                                y=[nifty_peak_val, nifty_trough_val],
                                                                mode='markers+text',
                                                                marker=dict(color=['red', 'green'], size=10),
                                                                text=['Peak', 'Trough'],
                                                                textposition="top center",name="Identified Peak/Trough"))
                        st.plotly_chart(fig_corr_nifty, use_container_width=True)
                st.markdown("---")
        else:
            st.info(f"No market correction data found in '{MARKET_CORRECTIONS_CSV_PATH}'. Please add/manage entries in the 'Data Management Hub'.")
    st.markdown("---")

    with st.expander("Market News (via Google News)", expanded=False):
        st.subheader("News Related to Market Conditions")
        search_query_news = st.text_input(
            "Keywords for news search:",
            value="Indian stock market sentiment OR Nifty PE OR FII DII activity India",
            key="market_news_query_mc_v2"
        )
        num_news_articles = st.number_input(
            "Number of news articles to fetch:",
            min_value=1, max_value=10, value=3, step=1,
            key="market_news_num_mc_v2"
        )
        if st.button("Fetch Market News", key="fetch_market_news_btn_mc_v2"):
            if not search_query_news:
                st.warning("Please enter keywords to search for news.")
            else:
                with st.spinner(f"Fetching news for '{search_query_news}'..."):
                    try:
                        google_news = GNews(language='en', country='IN', period='7d', max_results=num_news_articles)
                        json_resp = google_news.get_news(search_query_news)
                        if json_resp:
                            st.write(f"**Recent News for '{search_query_news}':**")
                            for item in json_resp:
                                st.markdown(f"**[{item.get('title', 'No Title')}]({item.get('url')})**")
                                st.caption(f"Source: {item.get('publisher', {}).get('title', 'N/A')} | Published: {item.get('published date', 'N/A')}")
                                st.markdown("---")
                        else:
                            st.info(f"No news found for '{search_query_news}'. Try different keywords.")
                    except Exception as e:
                        st.error(f"Error fetching news via gnews: {e}")
    st.markdown("---")

    with st.expander("Other Index P/E Ratios (Placeholder)", expanded=False):
        st.info("Functionality to analyze P/E for other sectoral and broad market indices will be added here (as part of Sectoral Analysis).")
    
    with st.expander("Market Cap to GDP Ratio (Placeholder)", expanded=False):
        st.info("Data and analysis for the Market Cap to GDP ratio (Buffett Indicator) will be displayed here (as part of Economic Indicators).")
    
    with st.expander("Banking Sector Sentiment (Placeholder)", expanded=False):
        st.info("Indicators related to banking sector health and credit growth will be presented here (part of Banking Sector Health).")