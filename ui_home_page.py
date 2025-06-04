# ui_home_page.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
from data_fetcher import get_stock_info, get_stock_history # get_financial_statements was not used in this snippet
from utils import (
    safe_get, format_large_number, convert_df_to_csv, parse_date_from_string,
    calculate_rsi, calculate_macd, calculate_bollinger_bands
)
import yfinance as yf 
from datetime import datetime, timezone, date 
import urllib.parse 
# import re # re was not used in this specific file snippet, remove if not used elsewhere in it

def render_financial_trend_charts(df, period_type):
    if df.empty:
        st.info(f"No {period_type.lower()} financial data available for trend charts.")
        return
    
    # Select relevant metrics for charts
    revenue = df.loc['Total Revenue'] if 'Total Revenue' in df.index else None
    net_income = df.loc['Net Income'] if 'Net Income' in df.index else None
    operating_income = df.loc['Operating Income'] if 'Operating Income' in df.index else None
    
    if revenue is not None or net_income is not None or operating_income is not None:
        for metric_data, metric_name in [
            (revenue, 'Revenue'),
            (net_income, 'Net Income'),
            (operating_income, 'Operating Income')
        ]:
            if metric_data is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[d.strftime('%Y-%m-%d') if isinstance(d, (datetime, pd.Timestamp)) else str(d) for d in metric_data.index],
                    y=metric_data.values,
                    name=metric_name
                ))
                fig.update_layout(
                    title=f"{period_type} {metric_name} Trend",
                    xaxis_title="Period",
                    yaxis_title="Amount",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue, net income, or operating income data available for charts.")

def render_home_page():
    st.title("Stock Deep Dive 🔎")
    user_input = st.text_input("Enter Stock Name or Ticker (e.g., RELIANCE or RELIANCE.NS):", "").upper()
    determined_ticker = None 
    stock_obj = None 
    if user_input: 
        # ... (ticker determination logic remains the same) ...
        if '.' in user_input: determined_ticker = user_input
        else:
            # st.write(f"Searching for '{user_input}' (assuming .NS if not specified)...") # Optional: Keep if you like the feedback
            ticker_ns = user_input + ".NS"; temp_info_ns = get_stock_info(ticker_ns)
            if temp_info_ns and safe_get(temp_info_ns, 'longName') != 'N/A':
                determined_ticker = ticker_ns
                st.info(f"Displaying data for {determined_ticker} (NSE). To specify BSE, use '.BO' suffix.")
            else:
                ticker_bo = user_input + ".BO"; temp_info_bo = get_stock_info(ticker_bo)
                if temp_info_bo and safe_get(temp_info_bo, 'longName') != 'N/A':
                    determined_ticker = ticker_bo
                    st.info(f"Displaying data for {determined_ticker} (BSE).")
                else: 
                    st.error(f"Could not find '{user_input}' with .NS or .BO suffix. Please check the ticker or try specifying the exchange (e.g., RELIANCE.NS).")
                    determined_ticker = None 
        
        if determined_ticker: 
            try:
                stock_obj = yf.Ticker(determined_ticker) 
                stock_info = get_stock_info(determined_ticker) 
                if not stock_info or safe_get(stock_info, 'longName') == 'N/A':
                    st.error(f"Could not retrieve valid information for {determined_ticker}. Please check the ticker again.")
                    return 
            except Exception as e: st.error(f"Error initializing yfinance Ticker or fetching info for {determined_ticker}: {e}"); return

            # ... (Company Overview & Key Metrics section remains the same) ...
            st.subheader(f"Showing results for: {determined_ticker}")
            st.markdown("---"); st.header("Company Overview & Key Metrics")
            st.write(f"**{safe_get(stock_info, 'longName')}** ({safe_get(stock_info, 'symbol')})")
            st.write(f"**Sector:** {safe_get(stock_info, 'sector')} | **Industry:** {safe_get(stock_info, 'industry')}")
            st.write(f"**Exchange:** {safe_get(stock_info, 'exchangeName')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                current_price_val = safe_get(stock_info, 'currentPrice', safe_get(stock_info, 'regularMarketPreviousClose'))
                st.metric(label="Current Price", value=f"{current_price_val}" if current_price_val != 'N/A' and current_price_val is not None else "N/A")
                st.metric(label="Market Cap", value=f"{format_large_number(safe_get(stock_info, 'marketCap', None))}") # Pass None for missing numeric
                st.metric(label="Volume", value=f"{safe_get(stock_info, 'volume', 0):,}")
                st.metric(label="Avg. Volume (10D)", value=f"{safe_get(stock_info, 'averageDailyVolume10Day', 0):,}")
            with col2:
                st.metric(label="Day High", value=f"{safe_get(stock_info, 'dayHigh', None)}")
                st.metric(label="Day Low", value=f"{safe_get(stock_info, 'dayLow', None)}")
                st.metric(label="52-Week High", value=f"{safe_get(stock_info, 'fiftyTwoWeekHigh', None)}")
                st.metric(label="52-Week Low", value=f"{safe_get(stock_info, 'fiftyTwoWeekLow', None)}")
            with col3:
                trailing_pe = safe_get(stock_info, 'trailingPE', None); st.metric(label="P/E Ratio (TTM)", value=f"{trailing_pe:.2f}" if isinstance(trailing_pe, float) else "N/A")
                forward_pe = safe_get(stock_info, 'forwardPE', None); st.metric(label="Forward P/E", value=f"{forward_pe:.2f}" if isinstance(forward_pe, float) else "N/A")
                price_to_book = safe_get(stock_info, 'priceToBook', None); st.metric(label="P/B Ratio", value=f"{price_to_book:.2f}" if isinstance(price_to_book, float) else "N/A")
                dividend_yield = safe_get(stock_info, 'dividendYield', None); st.metric(label="Dividend Yield", value=f"{dividend_yield*100:.2f}%" if isinstance(dividend_yield, float) else "N/A")
            
            # ... (Business Summary, Calendar, Shareholding sections remain the same) ...
            st.subheader("Business Summary"); st.write(safe_get(stock_info, 'longBusinessSummary', 'No summary available.')); st.markdown("---")
            st.header("Company Calendar / Events")
            try:
                calendar_data = stock_obj.calendar
                if isinstance(calendar_data, pd.DataFrame) and not calendar_data.empty:
                    calendar_display_df = calendar_data.T.copy() 
                    for col_name_cal in calendar_display_df.columns: 
                        if not calendar_display_df[col_name_cal].empty:
                            raw_date_val = calendar_display_df[col_name_cal].iloc[0]; parsed_dates = []
                            date_values_to_process = raw_date_val if isinstance(raw_date_val, list) else [raw_date_val]
                            for d_item in date_values_to_process:
                                if isinstance(d_item, (datetime, pd.Timestamp, date)): parsed_dates.append(d_item.strftime('%B %d, %Y'))
                                elif isinstance(d_item, str): 
                                    parsed_d = parse_date_from_string(d_item)
                                    if parsed_d: parsed_dates.append(parsed_d.strftime('%B %d, %Y'))
                                    else: parsed_dates.append(d_item) 
                            if parsed_dates: calendar_display_df[col_name_cal] = " to ".join(parsed_dates) if len(parsed_dates) > 1 else parsed_dates[0]
                            elif isinstance(raw_date_val, (int, float)): calendar_display_df[col_name_cal] = f"{raw_date_val:,.2f}" if isinstance(raw_date_val, float) else f"{raw_date_val:,}"
                    st.dataframe(calendar_display_df)
                elif isinstance(calendar_data, dict) and calendar_data: 
                    for event_type, event_details in calendar_data.items():
                        display_value = ""; event_details_list = event_details if isinstance(event_details, list) else [event_details]; formatted_details = []
                        for item_detail in event_details_list:
                            if isinstance(item_detail, (datetime, pd.Timestamp, date)): formatted_details.append(item_detail.strftime('%B %d, %Y'))
                            elif isinstance(item_detail, str):
                                parsed_d = parse_date_from_string(item_detail)
                                if parsed_d: formatted_details.append(parsed_d.strftime('%B %d, %Y'))
                                else: formatted_details.append(item_detail)
                            elif isinstance(item_detail, (int, float)): formatted_details.append(f"{item_detail:,.2f}" if isinstance(item_detail, float) else f"{item_detail:,}")
                            else: formatted_details.append(str(item_detail))
                        display_value = ", ".join(formatted_details) if len(formatted_details) > 1 else (formatted_details[0] if formatted_details else str(event_details))
                        st.markdown(f"**{event_type.replace('_', ' ').title()}:** {display_value}")
                else: st.info("No calendar event data available for this stock.")
            except Exception as e: st.info(f"Could not retrieve or process calendar data: {e}")
            st.markdown("---")
            st.header("Shareholding Overview")
            insider_percentage_float = safe_get(stock_info, 'heldPercentInsiders', None); institution_percentage_float = safe_get(stock_info, 'heldPercentInstitutions', None)
            share_labels, share_values, valid_data_found = [], [], False
            if isinstance(insider_percentage_float, float): share_labels.append("Insiders"); share_values.append(insider_percentage_float * 100); valid_data_found = True
            if isinstance(institution_percentage_float, float): share_labels.append("Institutions"); share_values.append(institution_percentage_float * 100); valid_data_found = True
            if valid_data_found:
                total_known_percentage = sum(share_values)
                if total_known_percentage <= 100.5 and total_known_percentage > 0 : # Avoid adding if 100% is already covered
                     if 100 - total_known_percentage > 0.5 : # Only add if public is a meaningful slice
                        share_labels.append("Public / Other"); share_values.append(max(0, 100 - total_known_percentage)) 
                if share_values:
                    fig_shareholding = px.pie(names=share_labels, values=share_values, title="Shareholding Pattern"); fig_shareholding.update_traces(textposition='inside', textinfo='percent+label'); st.plotly_chart(fig_shareholding)
                else: st.info("Shareholding percentage data could not be visualized.")
            else: st.info("Shareholding percentage data (insiders/institutions) not available for a chart.")
            
            # ... (Price History & Technicals section remains largely the same) ...
            st.markdown("---")
            st.header("Price History & Technicals")
            control_cols_row1 = st.columns(3); interval_map = {"1 min": "1m", "5 mins": "5m", "15 mins": "15m", "30 mins": "30m", "1 hour": "1h", "1 day": "1d", "1 week": "1wk", "1 month": "1mo"}; display_intervals = list(interval_map.keys())
            with control_cols_row1[0]:
                selected_display_interval = st.selectbox("Interval:", display_intervals, index=display_intervals.index("1 day"), key=f"chart_interval_{determined_ticker}"); yf_interval = interval_map[selected_display_interval]
            with control_cols_row1[1]: period_qty = st.number_input("Period Value:", min_value=1, value=1, step=1, key=f"period_qty_{determined_ticker}")
            with control_cols_row1[2]:
                period_unit_options = ["days", "months", "years", "max"]; default_unit_index = 0 
                if yf_interval == "1d": default_unit_index = period_unit_options.index("years")
                elif yf_interval in ["1wk", "1mo"]: default_unit_index = period_unit_options.index("years")
                selected_period_unit = st.selectbox("Period Unit:", period_unit_options, index=default_unit_index, key=f"period_unit_{determined_ticker}")
            if selected_period_unit == "max": yf_period = "max"
            else: unit_suffix = {'days': 'd', 'months': 'mo', 'years': 'y'}; yf_period = f"{period_qty}{unit_suffix[selected_period_unit]}"
            st.markdown("<h6>Technical Indicators & Overlays</h6>", unsafe_allow_html=True)
            indicator_controls_main_cols = st.columns([1,4])
            with indicator_controls_main_cols[0]:
                if st.button("Refresh", key=f"refresh_chart_tech_{determined_ticker}"): st.rerun()
            with indicator_controls_main_cols[1]:
                indicator_grid = st.columns(4) 
                selected_smas = []; sma_options = {"SMA 20": 20, "SMA 50": 50, "SMA 200": 200}
                with indicator_grid[0]:
                    st.markdown("**SMAs**")
                    for sma_label, sma_val in sma_options.items():
                        if st.checkbox(sma_label, key=f"sma_{sma_val}_{determined_ticker}"): selected_smas.append(sma_val)
                selected_dmas = []; dma_options = {"DMA 50": 50, "DMA 100": 100, "DMA 200": 200}
                with indicator_grid[1]:
                    st.markdown("**DMAs**")
                    dma_displacement = st.number_input("Displacement:", min_value=-50, max_value=50, value=10, step=1, key=f"dma_disp_{determined_ticker}")
                    for dma_label, dma_val in dma_options.items():
                        if st.checkbox(dma_label, key=f"dma_{dma_val}_{determined_ticker}"): selected_dmas.append(dma_val)
                with indicator_grid[2]:
                    st.markdown("**Bollinger Bands**")
                    show_bbands = st.checkbox("Show BB", key=f"bbands_show_{determined_ticker}")
                    bb_window = st.number_input("BB Window:", min_value=5, value=20, step=1, key=f"bb_window_{determined_ticker}", disabled=not show_bbands)
                    bb_std_dev = st.number_input("BB Std Dev:", min_value=1.0, value=2.0, step=0.1, format="%.1f", key=f"bb_std_{determined_ticker}", disabled=not show_bbands)
                with indicator_grid[3]:
                    st.markdown("**Oscillators**")
                    show_rsi = st.checkbox("Show RSI", key=f"rsi_show_{determined_ticker}")
                    rsi_period = st.number_input("RSI Period:", min_value=2, value=14, step=1, key=f"rsi_period_{determined_ticker}", disabled=not show_rsi)
                    show_macd = st.checkbox("Show MACD", key=f"macd_show_{determined_ticker}")
                    if show_macd: 
                        macd_fast = st.number_input("Fast:", min_value=1, value=12, step=1, key=f"macd_fast_{determined_ticker}")
                        macd_slow = st.number_input("Slow:", min_value=1, value=26, step=1, key=f"macd_slow_{determined_ticker}")
                        macd_signal = st.number_input("Signal:", min_value=1, value=9, step=1, key=f"macd_signal_{determined_ticker}")
                    else: macd_fast, macd_slow, macd_signal = 12, 26, 9 
            display_period_string = f"{period_qty} {selected_period_unit}" if selected_period_unit != "max" else "Max"; chart_title = f"{determined_ticker} ({selected_display_interval} - {display_period_string})"
            with st.spinner(f"Loading chart data: {selected_display_interval} for {display_period_string}..."): stock_history_df = get_stock_history(determined_ticker, period=yf_period, interval=yf_interval)
            if stock_history_df is not None and not stock_history_df.empty:
                num_subplots = 2; row_heights = [0.7, 0.3] 
                if show_rsi: num_subplots += 1
                if show_macd: num_subplots += 1
                if num_subplots == 2: row_heights = [0.7, 0.3]
                elif num_subplots == 3: row_heights = [0.6, 0.2, 0.2] 
                elif num_subplots == 4: row_heights = [0.5, 0.17, 0.17, 0.16] 
                fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)
                current_row = 1
                fig.add_trace(go.Scatter(x=stock_history_df.index, y=stock_history_df['Close'], name='Close', line=dict(color='#636EFA')), row=1, col=1)
                if show_bbands:
                    upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(stock_history_df['Close'], window=bb_window, num_std_dev=bb_std_dev)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=upper_bb, name=f'Upper BB({bb_window},{bb_std_dev:.1f})', line=dict(color='rgba(150,150,150,0.5)', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=middle_bb, name=f'Middle BB({bb_window})', line=dict(color='rgba(150,150,150,0.5)', dash='dot')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=lower_bb, name=f'Lower BB({bb_window},{bb_std_dev:.1f})', line=dict(color='rgba(150,150,150,0.5)', dash='dash'),fill='tonexty', fillcolor='rgba(150,150,150,0.1)'), row=1, col=1)
                for sma_val in selected_smas:
                    if len(stock_history_df) >= sma_val: stock_history_df[f'SMA {sma_val}'] = stock_history_df['Close'].rolling(window=sma_val).mean(); fig.add_trace(go.Scatter(x=stock_history_df.index, y=stock_history_df[f'SMA {sma_val}'], name=f'SMA {sma_val}', opacity=0.7), row=1, col=1)
                for dma_val in selected_dmas:
                    if len(stock_history_df) >= dma_val: sma_temp = stock_history_df['Close'].rolling(window=dma_val).mean(); stock_history_df[f'DMA {dma_val}'] = sma_temp.shift(dma_displacement); fig.add_trace(go.Scatter(x=stock_history_df.index, y=stock_history_df[f'DMA {dma_val}'], name=f'DMA {dma_val}(D:{dma_displacement})', opacity=0.7, line=dict(dash='dot')), row=1, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1); current_row += 1
                fig.add_trace(go.Bar(x=stock_history_df.index, y=stock_history_df['Volume'], name='Volume', marker_color='rgba(119,119,119,0.5)'), row=current_row, col=1)
                fig.update_yaxes(title_text="Volume", row=current_row, col=1); 
                if num_subplots == current_row: fig.update_xaxes(title_text="Date", row=current_row, col=1)
                else: fig.update_xaxes(showticklabels=False, row=current_row, col=1)
                current_row += 1
                if show_rsi:
                    stock_history_df['RSI'] = calculate_rsi(stock_history_df['Close'], window=rsi_period)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=stock_history_df['RSI'], name=f'RSI({rsi_period})', line=dict(color='orange')), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
                    fig.update_yaxes(title_text="RSI", range=[0,100], row=current_row, col=1)
                    if num_subplots == current_row: fig.update_xaxes(title_text="Date", row=current_row, col=1)
                    else: fig.update_xaxes(showticklabels=False, row=current_row, col=1)
                    current_row += 1
                if show_macd:
                    macd_line, signal_line, macd_hist = calculate_macd(stock_history_df['Close'], macd_slow, macd_fast, macd_signal)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=macd_line, name='MACD', line=dict(color='purple')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=stock_history_df.index, y=signal_line, name='Signal', line=dict(color='cyan')), row=current_row, col=1)
                    colors = ['green' if val >= 0 else 'red' for val in macd_hist]; fig.add_trace(go.Bar(x=stock_history_df.index, y=macd_hist, name='MACD Hist', marker_color=colors), row=current_row, col=1)
                    fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                    if num_subplots == current_row: fig.update_xaxes(title_text="Date", row=current_row, col=1)
                chart_height = 350 + (150 * (num_subplots -1)); fig.update_layout(title_text=chart_title, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", height=chart_height, margin=dict(l=50, r=20, t=80, b=20))
                for i in range(1, num_subplots): fig.update_xaxes(showticklabels=False, row=i, col=1)
                fig.update_xaxes(showticklabels=True, row=num_subplots, col=1)
                st.plotly_chart(fig, use_container_width=True)
                csv_price = convert_df_to_csv(stock_history_df); st.download_button(label="Download Data as CSV", data=csv_price, file_name=f"{determined_ticker}_data.csv", mime='text/csv', key=f"download_data_{determined_ticker}")
            else: st.warning(f"Could not retrieve history for {determined_ticker} ({yf_period}, {yf_interval}).")
            st.markdown("---")
            st.header("Peer Comparison")
            peer_tickers_str = st.text_input(f"Enter peer tickers for {determined_ticker} (comma-separated, e.g., STOCK1.NS,STOCK2.BO):", key=f"peers_{determined_ticker}")
            compare_button_key = f"compare_peers_btn_{determined_ticker}"
            
            if st.button("Compare Peers", key=compare_button_key):
                if peer_tickers_str:
                    peer_tickers = [ticker.strip().upper() for ticker in peer_tickers_str.split(',') if ticker.strip()]
                    all_tickers_for_comparison = [determined_ticker] + peer_tickers
                    comparison_data = []
                    metrics_to_compare = {
                        'Symbol': 'symbol', 'Company Name': 'longName', 'Market Cap': 'marketCap', 
                        'P/E (TTM)': 'trailingPE', 'P/B Ratio': 'priceToBook', 'Div Yield': 'dividendYield', 
                        'Revenue Growth (YoY)': 'revenueGrowth', 'EPS (TTM)': 'trailingEps', 
                        'ROE': 'returnOnEquity', 'Debt/Equity': 'debtToEquity'
                    }
                    
                    with st.spinner("Fetching peer data..."):
                        for ticker_symbol_comp in all_tickers_for_comparison:
                            info = get_stock_info(ticker_symbol_comp) 
                            data_row = {'Ticker': ticker_symbol_comp}
                            if info:
                                for display_key, yf_key in metrics_to_compare.items():
                                    if yf_key == 'symbol': continue # 'Ticker' column already has this
                                    data_row[display_key] = safe_get(info, yf_key, default=None) # Use None for missing
                            else:
                                st.warning(f"Could not fetch data for peer: {ticker_symbol_comp}")
                                for display_key, yf_key in metrics_to_compare.items():
                                    if yf_key != 'symbol': data_row[display_key] = None # Use None for missing data
                            comparison_data.append(data_row)
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        # Columns that should be numeric
                        numeric_cols = ['Market Cap', 'P/E (TTM)', 'P/B Ratio', 'Div Yield', 
                                        'Revenue Growth (YoY)', 'EPS (TTM)', 'ROE', 'Debt/Equity']
                        
                        for col in numeric_cols:
                            if col in df_comparison.columns:
                                df_comparison[col] = pd.to_numeric(df_comparison[col], errors='coerce')

                        # Create a display DataFrame for formatting
                        df_display = df_comparison.copy()

                        # Apply specific string formatting for display
                        if 'Market Cap' in df_display.columns:
                            df_display['Market Cap'] = df_display['Market Cap'].apply(
                                lambda x: format_large_number(x, default='N/A') if pd.notna(x) else 'N/A'
                            )
                        
                        percent_cols = ['Div Yield', 'Revenue Growth (YoY)', 'ROE']
                        for col in percent_cols:
                            if col in df_display.columns:
                                df_display[col] = df_display[col].apply(
                                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A'
                                )
                        
                        float_str_cols = ['P/E (TTM)', 'P/B Ratio', 'EPS (TTM)', 'Debt/Equity']
                        for col in float_str_cols:
                            if col in df_display.columns:
                                df_display[col] = df_display[col].apply(
                                    lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
                                )
                        
                        display_cols_ordered = ['Ticker'] + [col for col in metrics_to_compare.keys() if col != 'Symbol']
                        
                        # Ensure all columns are present and in order for the final display DataFrame
                        final_display_df = pd.DataFrame(columns=display_cols_ordered)
                        for col in display_cols_ordered:
                            if col in df_display.columns:
                                final_display_df[col] = df_display[col]
                            else: # Should not happen if metrics_to_compare is source of truth
                                final_display_df[col] = 'N/A' 
                        
                        st.dataframe(final_display_df.set_index('Ticker'))
                    else: 
                        st.info("No data fetched for comparison.")
                else: 
                    st.warning("Please enter some peer tickers to compare.")

            # ... (Concall Notes, Financial Statements, Dividend, Splits, News sections remain the same) ...
            st.markdown("---")
            st.header("Concall Notes & Links")
            concall_notes_key = f"concall_notes_{determined_ticker}" 
            if concall_notes_key not in st.session_state: st.session_state[concall_notes_key] = "" 
            concall_text = st.text_area("Enter your notes or summary from concalls here:", value=st.session_state[concall_notes_key], height=150, key=f"text_area_concall_{determined_ticker}")
            st.session_state[concall_notes_key] = concall_text
            search_query_base = safe_get(stock_info, 'longName', determined_ticker)
            encoded_query_transcript = urllib.parse.quote_plus(f"{search_query_base} earnings call transcript"); encoded_query_ir = urllib.parse.quote_plus(f"{search_query_base} investor relations")
            button_cols = st.columns(2)
            with button_cols[0]: st.link_button("Search Transcript (Google)", f"https://www.google.com/search?q={encoded_query_transcript}")
            with button_cols[1]: st.link_button("Search Investor Relations (Google)", f"https://www.google.com/search?q={encoded_query_ir}")
            st.subheader("Generate Transcript Highlights (Temporarily Disabled)")
            st.info("The automated transcript summarization feature is currently disabled due to NLTK environment issues. We can revisit this later. You can still use the manual notes section above.")
            st.caption("Automatic concall fetching/summarization is not supported by yfinance directly."); st.markdown("---")
            st.header("Financial Statements & Trends")
            financial_statements_data = {
                "annual_financials": stock_obj.financials if hasattr(stock_obj, 'financials') else pd.DataFrame(),
                "quarterly_financials": stock_obj.quarterly_financials if hasattr(stock_obj, 'quarterly_financials') else pd.DataFrame(),
                "annual_balance_sheet": stock_obj.balance_sheet if hasattr(stock_obj, 'balance_sheet') else pd.DataFrame(),
                "quarterly_balance_sheet": stock_obj.quarterly_balance_sheet if hasattr(stock_obj, 'quarterly_balance_sheet') else pd.DataFrame(),
                "annual_cashflow": stock_obj.cashflow if hasattr(stock_obj, 'cashflow') else pd.DataFrame(),
                "quarterly_cashflow": stock_obj.quarterly_cashflow if hasattr(stock_obj, 'quarterly_cashflow') else pd.DataFrame(),
            }
            if financial_statements_data:
                annual_data_available = not financial_statements_data["annual_financials"].empty
                quarterly_data_available = not financial_statements_data["quarterly_financials"].empty
                tab_titles = []
                if annual_data_available: tab_titles.append("Annual Data & Trends")
                if quarterly_data_available: tab_titles.append("Quarterly Data & Trends")
                if not tab_titles: st.info("No annual or quarterly financial statement data available to display.")
                else:
                    tabs = st.tabs(tab_titles); current_tab_index = 0
                    if annual_data_available:
                        with tabs[current_tab_index]:
                            st.subheader("Annual Financial Statements")
                            statement_type_annual = st.selectbox("Select Annual Statement:",["Income Statement", "Balance Sheet", "Cash Flow"],key=f"statement_annual_{determined_ticker}")
                            df_annual_orig = None; annual_key_map = {"Income Statement": "annual_financials", "Balance Sheet": "annual_balance_sheet", "Cash Flow": "annual_cashflow"}
                            selected_annual_key = annual_key_map.get(statement_type_annual)
                            if selected_annual_key and not financial_statements_data[selected_annual_key].empty: df_annual_orig = financial_statements_data[selected_annual_key]
                            if df_annual_orig is not None:
                                df_annual_display = df_annual_orig.copy(); df_annual_display.columns = [col.strftime('%B %d, %Y') if isinstance(col, (pd.Timestamp, datetime)) else col for col in df_annual_display.columns]
                                st.dataframe(df_annual_display); csv_annual = convert_df_to_csv(df_annual_orig)
                                st.download_button(label=f"Download Annual {statement_type_annual} as CSV", data=csv_annual, file_name=f"{determined_ticker}_Annual_{statement_type_annual.replace(' ', '_')}.csv", mime='text/csv', key=f"download_annual_{statement_type_annual.replace(' ', '_')}_{determined_ticker}")
                                if statement_type_annual == "Income Statement":
                                    with st.expander("Show Annual Financial Trend Charts", expanded=False): render_financial_trend_charts(df_annual_orig, "Annual")
                            else: st.info(f"Annual {statement_type_annual} data not available.")
                        current_tab_index +=1
                    if quarterly_data_available:
                        with tabs[current_tab_index]: 
                            st.subheader("Quarterly Financial Statements")
                            statement_type_quarterly = st.selectbox("Select Quarterly Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"], key=f"statement_quarterly_{determined_ticker}")
                            df_quarterly_orig = None; quarterly_key_map = {"Income Statement": "quarterly_financials", "Balance Sheet": "quarterly_balance_sheet", "Cash Flow": "quarterly_cashflow"}
                            selected_quarterly_key = quarterly_key_map.get(statement_type_quarterly)
                            if selected_quarterly_key and not financial_statements_data[selected_quarterly_key].empty: df_quarterly_orig = financial_statements_data[selected_quarterly_key]
                            if df_quarterly_orig is not None:
                                df_quarterly_display = df_quarterly_orig.copy(); df_quarterly_display.columns = [col.strftime('%B %d, %Y') if isinstance(col, (pd.Timestamp, datetime)) else col for col in df_quarterly_display.columns]
                                st.dataframe(df_quarterly_display); csv_quarterly = convert_df_to_csv(df_quarterly_orig)
                                st.download_button(label=f"Download Quarterly {statement_type_quarterly} as CSV", data=csv_quarterly, file_name=f"{determined_ticker}_Quarterly_{statement_type_quarterly.replace(' ', '_')}.csv", mime='text/csv', key=f"download_quarterly_{statement_type_quarterly.replace(' ', '_')}_{determined_ticker}")
                                if statement_type_quarterly == "Income Statement":
                                     with st.expander("Show Quarterly Financial Trend Charts", expanded=False): render_financial_trend_charts(df_quarterly_orig, "Quarterly")
                            else: st.info(f"Quarterly {statement_type_quarterly} data not available.")
            st.markdown("---")
            st.header("Dividend History")
            dividends_df_orig = stock_obj.dividends if hasattr(stock_obj, 'dividends') else pd.DataFrame()
            if not dividends_df_orig.empty: st.dataframe(dividends_df_orig.sort_index(ascending=False).rename(index=lambda x: x.strftime('%B %d, %Y')))
            else: st.info("No dividend data available for this stock.")
            st.markdown("---")
            st.header("Corporate Actions (Splits)")
            splits_df_orig = stock_obj.splits if hasattr(stock_obj, 'splits') else pd.DataFrame()
            if not splits_df_orig.empty: st.dataframe(splits_df_orig.sort_index(ascending=False).rename(index=lambda x: x.strftime('%B %d, %Y')))
            else: st.info("No stock split data available for this stock.")
            st.markdown("---")
            st.header("Recent News")
            news_items = stock_obj.news if hasattr(stock_obj, 'news') else []
            displayed_news_count = 0
            if news_items:
                for item_wrapper in news_items[:15]: 
                    item_content = item_wrapper.get('content', {}); title = item_content.get('title')
                    if title : 
                        click_url_data = item_content.get('clickThroughUrl', {}); link = click_url_data.get('url')
                        if not link: canonical_url_data = item_content.get('canonicalUrl', {}); link = canonical_url_data.get('url')
                        provider_data = item_content.get('provider', {}); publisher = provider_data.get('displayName', 'N/A')
                        pub_date_str_val = item_content.get('pubDate')
                        if pub_date_str_val:
                            try:
                                if pub_date_str_val.endswith('Z'): pub_date_str_val = pub_date_str_val[:-1]
                                dt_object = datetime.fromisoformat(pub_date_str_val).replace(tzinfo=timezone.utc)
                                publish_time_display = dt_object.strftime('%B %d, %Y') 
                            except ValueError: publish_time_display = pub_date_str_val 
                        else: publish_time_display = 'N/A'
                        summary = item_content.get('summary', '') 
                        if link: st.subheader(f"[{title}]({link})")
                        else: st.subheader(title)
                        st.caption(f"Publisher: {publisher} | Published: {publish_time_display}")
                        if summary: st.write(summary)
                        st.markdown("---"); displayed_news_count += 1
                        if displayed_news_count >= 7: break 
            if displayed_news_count == 0: st.info("No recent news with titles available for this stock.")
            
            # Simplified portfolio add button (local session state only, not Firebase)
            # If integration with the main portfolio_page is desired, this needs to call add_stock_to_firestore
            if st.button("Add to Quick Watchlist", key=f"add_to_quick_watchlist_{determined_ticker}"):
                if 'quick_watchlist' not in st.session_state: 
                    st.session_state.quick_watchlist = {}
                if determined_ticker not in st.session_state.quick_watchlist:
                    st.session_state.quick_watchlist[determined_ticker] = {
                        'name': safe_get(stock_info, 'longName', determined_ticker),
                        'added_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success(f"{determined_ticker} added to your Quick Watchlist!")
                else: 
                    st.info(f"{determined_ticker} is already in your Quick Watchlist.")