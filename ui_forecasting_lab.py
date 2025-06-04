# ui_forecasting_lab.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.holtwinters import Holt 
from utils import load_csv_data # Use the generic loader

FII_DATA_CSV_PATH = "historical_fii_data.csv" 

def render_forecasting_lab_page():
    st.title("Forecasting Lab 🔬")
    st.markdown("---")

    st.info("""
    **Welcome to the Forecasting Lab!** This section is for experimental time-series projections. 
    Please remember that these are based on simple statistical models and past data only. 
    They do **NOT** account for fundamental factors, news, or other market dynamics and should **NOT** be used as financial advice or reliable predictions.
    """)

    # --- FII Flow Projection ---
    st.header("Simple FII Flow Projection (Experimental)")
    
    combined_fii_df_proj = load_csv_data(
        FII_DATA_CSV_PATH, 
        default_cols=['Date', 'FII_Total_Net_Investment'], 
        parse_dates_cols=['Date'],
        error_level="warning" 
    )
    
    if not combined_fii_df_proj.empty and 'FII_Total_Net_Investment' in combined_fii_df_proj.columns:
        combined_fii_df_proj['Date'] = pd.to_datetime(combined_fii_df_proj['Date'], errors='coerce')
        combined_fii_df_proj = combined_fii_df_proj.dropna(subset=['Date'])
        combined_fii_df_proj = combined_fii_df_proj.sort_values(by="Date").reset_index(drop=True)
        
        fii_series_for_proj = combined_fii_df_proj.set_index('Date')['FII_Total_Net_Investment'].dropna()
        
        if len(fii_series_for_proj) > 12 : 
            projection_months = st.number_input("Months to Project FII Flow:", min_value=1, max_value=24, value=6, step=1, key="fii_proj_lab_months_final")
            
            if st.button("Generate FII Flow Projection", key="gen_fii_proj_lab_final"):
                with st.spinner("Generating FII flow projection..."):
                    try:
                        fii_series_numeric = fii_series_for_proj.astype(float)
                        fii_series_resampled = fii_series_numeric.asfreq('MS', method='pad') 

                        if fii_series_resampled.isnull().any():
                            st.warning("Data contains NaNs after resampling to monthly frequency, attempting to fill. Projection might be less accurate.")
                            fii_series_resampled = fii_series_resampled.fillna(method='pad').fillna(method='bfill') # Fill any remaining NaNs

                        if not fii_series_resampled.empty and len(fii_series_resampled) > 1 : # Holt needs at least 2 obs
                            model = Holt(fii_series_resampled, initialization_method="estimated")
                            fit = model.fit(optimized=True) 
                            forecast_values = fit.forecast(projection_months)
                            
                            # forecast_dates = pd.date_range(start=fii_series_resampled.index.max() + pd.DateOffset(months=1), periods=projection_months, freq='MS') 
                            # Using forecast_values.index which Holt's model should provide correctly
                            forecast_df = pd.DataFrame({'Date': forecast_values.index, 'Projected_FII_Flow': forecast_values.values})

                            fig_proj = go.Figure()
                            fig_proj.add_trace(go.Scatter(x=fii_series_numeric.index, y=fii_series_numeric, mode='lines+markers', name='Historical FII Net Flow'))
                            fig_proj.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Projected_FII_Flow'], mode='lines+markers', name='Projected FII Net Flow', line=dict(dash='dash', color='red')))
                            fig_proj.update_layout(title_text="Historical and Projected FII Net Investment", xaxis_title="Date", yaxis_title="FII Net Investment (INR Cr)")
                            st.plotly_chart(fig_proj, use_container_width=True)
                            
                            st.write("Projected Values (INR Cr):")
                            display_forecast_df = forecast_df.copy()
                            display_forecast_df['Date'] = display_forecast_df['Date'].dt.strftime('%B %Y')
                            st.dataframe(display_forecast_df.set_index('Date'))
                        else:
                            st.error("Not enough data to fit Holt's model after resampling (or all values are NaN).")
                    except Exception as e:
                        st.error(f"Could not generate FII projection: {e}")
                        st.exception(e) 
        else:
            st.info("Not enough historical FII data (need more than 12 months from CSV) to generate a projection.")
    else:
        st.info("FII data not available or not in correct format for projection. Please ensure `historical_fii_data.csv` is present and correctly formatted via the 'Data Management Hub'.")

    st.markdown("---")
    st.header("Stock/Index Price Forecasting (Coming Soon)")
    st.info("Functionality to forecast stock/index prices using models like Prophet or ARIMA will be added here.")