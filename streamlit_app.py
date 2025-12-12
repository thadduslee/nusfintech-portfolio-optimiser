import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings

from backend.model import predict_next_day_volatility

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ML Smart Portfolio Rebalancer",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def predict_next_day_volatility_streamlit(ticker: str):
    try:
        vol, prices = predict_next_day_volatility(ticker, return_prices=True)
        return vol, prices, None
    except Exception as e:
        return None, None, str(e)

logo_col1, logo_col2, logo_col3 = st.sidebar.columns([1, 2, 1])
with logo_col2:
    try:
        st.image("https://img.icons8.com/fluency/96/bullish.png", width=80)
    except:
        st.write("📈")

st.sidebar.title("Smart Rebalancer")
st.sidebar.caption("Group 1: Dexun, Thaddus, Eron")
st.sidebar.divider()

# --- INPUTS ---
# 1. Tickers (Kept outside form so the main page updates immediately when you type)
default_tickers = "AAPL, NVDA, MSFT, BTC-USD, ETH-USD"
ticker_input = st.sidebar.text_area("Stocks (Comma Separated)", value=default_tickers)

# 2. Simulation Settings (Wrapped in a FORM to stop constant running when sliding)
with st.sidebar.form("simulation_settings"):
    st.write("### Simulation Settings")
    risk_free_rate = st.number_input("Risk Free Rate", value=0.04, step=0.01)

    num_simulations = st.slider(
        "Monte Carlo Simulations",
        min_value=1000,
        max_value=10000,
        value=2500,
        step=500,
        help="Higher = more accuracy but slower"
    )
    
    # The app will NOT rerun when you drag the slider. 
    # It only reruns when you click this button:
    st.form_submit_button("Apply Settings")

st.sidebar.divider()
st.sidebar.info("Use the Proposal tab to see methodology. Navigate to Live Optimizer to run the tool.")

tab_info, tab_app = st.tabs(["Proposal & Methodology", "Live Optimizer"])

with tab_info:
    st.title("ML-Powered Smart Portfolio Rebalancer")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Problem Statement")
        st.write("""
        Modern investors struggle to maintain balanced portfolios in volatile markets.
        Traditional tools are static and fail to adapt dynamically.
        """)

        st.subheader("Project Objective")
        st.write("""
        Build a machine learning-powered system that:
        1. Predicts next-day volatility using **XGBoost + EGARCH**
        2. Constructs a future covariance matrix
        3. Runs Monte Carlo optimization to maximize Sharpe Ratio
        """)

    with col2:
        st.info("""
        ### Key Features  
        - Multi-Asset Support  
        - EGARCH + XGBoost Hybrid  
        - Monte Carlo Simulation  
        - Interactive Portfolio Dashboard  
        """)

    st.divider()
    st.header("How It Works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 1. Feature Engineering")
        st.code("""
- Log Returns
- Lagged Volatilities
- Garman-Klass Volatility
- Volume Shock
""")

    with c2:
        st.markdown("### 2. Hybrid ML")
        st.info("""
EGARCH captures volatility clustering  
XGBoost corrects non-linear relationships  
""")

    with c3:
        st.markdown("### 3. Monte Carlo Optimization")
        st.success("""
Predict → Covariance → Optimal Weights  
""")

with tab_app:
    st.title("Live Portfolio Dashboard")

    stocks = [s.strip().upper() for s in ticker_input.split(",") if s.strip()]

    # Initialize weights in session state if not present or if tickers changed
    if "weights_df" not in st.session_state:
        eq = round(100 / len(stocks), 2) if stocks else 0
        st.session_state.weights_df = pd.DataFrame({
            "Asset": stocks,
            "Current Weight (%)": [eq] * len(stocks)
        })
    else:
        # Check if the number of stocks matches the dataframe
        if len(st.session_state.weights_df) != len(stocks):
            eq = round(100 / len(stocks), 2) if stocks else 0
            st.session_state.weights_df = pd.DataFrame({
                "Asset": stocks,
                "Current Weight (%)": [eq] * len(stocks)
            })
        else:
            # Just update the names in case they changed order/spelling
            st.session_state.weights_df["Asset"] = stocks

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("1. Define Current Allocation")
        edited_df = st.data_editor(
            st.session_state.weights_df,
            column_config={
                "Current Weight (%)": st.column_config.NumberColumn(
                    "Current Weight (%)",
                    min_value=0,
                    max_value=100,
                    step=0.1,
                    format="%.1f%%"
                )
            },
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        st.subheader("Validation")
        total_weight = edited_df["Current Weight (%)"].sum()
        st.metric("Total Allocation", f"{total_weight:.1f}%")

        ready = 99 <= total_weight <= 101

        if ready:
            st.success("Valid Sum")
        else:
            st.error("Weights must sum to ~100%")

    st.divider()

    # --- CALCULATION LOGIC (Only runs when clicked) ---
    if st.button("Run Prediction & Optimization", type="primary", disabled=not ready):

        if len(stocks) < 2:
            st.error("Enter at least 2 tickers.")
            st.stop()

        status = st.status("Processing Models...", expanded=True)
        progress = status.progress(0)

        predicted_vols = []
        valid_stocks = []
        price_series_list = []

        for i, stock in enumerate(stocks):
            status.write(f"Analyzing **{stock}** ...")
            vol, prices, err = predict_next_day_volatility_streamlit(stock)

            if vol is not None:
                predicted_vols.append(vol)
                valid_stocks.append(stock)
                price_series_list.append(prices.rename(stock))
            else:
                status.warning(f"{stock} skipped: {err}")

            progress.progress((i + 1) / len(stocks))

        if len(valid_stocks) < 2:
            status.update(label="Insufficient valid tickers", state="error")
            st.stop()

        status.write("Building Covariance Matrix...")

        price_df = pd.concat(price_series_list, axis=1)
        returns_df = np.log(price_df / price_df.shift(1)).dropna()
        corr = returns_df.corr()

        D = np.diag(np.array(predicted_vols) / 100)
        future_cov = D @ corr.values @ D

        status.write(f"Running {num_simulations} Monte Carlo Portfolios...")

        num_portfolios = num_simulations
        returns = []
        volatilities = []
        allocations = []

        arith_daily = returns_df.mean()
        arith_annual = arith_daily * 252
        geo_daily = (1 + returns_df).prod() ** (1 / len(returns_df)) - 1
        geo_annual = (1 + geo_daily) ** 252 - 1

        mean_annual = 0.5 * arith_annual + 0.5 * geo_annual
        mean_annual = mean_annual[valid_stocks]

        for _ in range(num_portfolios):
            w = np.random.random(len(valid_stocks))
            w /= np.sum(w)

            allocations.append(w)
            returns.append(np.dot(w, mean_annual))
            var = w.T @ future_cov @ w
            volatilities.append(np.sqrt(var))

        portfolio = pd.DataFrame({
            "Returns": returns,
            "Volatilities": volatilities
        })

        for i, s in enumerate(valid_stocks):
            portfolio[s] = [w[i] for w in allocations]

        sharpe = (portfolio["Returns"] - risk_free_rate) / portfolio["Volatilities"]
        best_idx = sharpe.idxmax()
        best_port = portfolio.iloc[best_idx]

        user_weights = dict(zip(edited_df["Asset"], edited_df["Current Weight (%)"]))
        rebal = []
        for s in valid_stocks:
            optimal = best_port[s] * 100
            current = user_weights.get(s, 0)
            diff = optimal - current

            if diff > 1:
                action = f"BUY (+{diff:.1f}%)"
            elif diff < -1:
                action = f"SELL ({diff:.1f}%)"
            else:
                action = "HOLD"

            rebal.append({
                "Asset": s,
                "Current Weight": f"{current:.2f}%",
                "Optimal Weight": f"{optimal:.2f}%",
                "Difference": f"{diff:+.2f}%",
                "Action": action,
                "Predicted Vol": f"{predicted_vols[valid_stocks.index(s)]:.2f}%"
            })

        # SAVE RESULTS TO SESSION STATE
        st.session_state['results'] = {
            'valid_stocks': valid_stocks,
            'predicted_vols': predicted_vols,
            'future_cov': future_cov,
            'portfolio': portfolio,
            'sharpe': sharpe,
            'best_port': best_port,
            'rebal': rebal,
            'user_weights': user_weights
        }

        status.update(label="Optimization Complete", state="complete", expanded=False)

    # --- DISPLAY LOGIC (Runs if results exist) ---
    if 'results' in st.session_state:
        results = st.session_state['results']
        valid_stocks = results['valid_stocks']
        predicted_vols = results['predicted_vols']
        future_cov = results['future_cov']
        portfolio = results['portfolio']
        sharpe = results['sharpe']
        best_port = results['best_port']
        rebal = results['rebal']
        user_weights = results['user_weights']

        st.subheader("Intermediate Model Outputs")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Predicted Annualized Volatility (%)**")
            st.dataframe(
                pd.DataFrame({"Asset": valid_stocks,
                              "Predicted Vol (%)": [f"{v:.4f}" for v in predicted_vols]}),
                hide_index=True
            )

        with c2:
            st.markdown("**Future Covariance Matrix**")
            st.dataframe(pd.DataFrame(future_cov, index=valid_stocks, columns=valid_stocks))

        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Sharpe Ratio", f"{sharpe.max():.2f}")
        m2.metric("Expected Return", f"{best_port['Returns'] * 100:.2f}%")
        m3.metric("Expected Volatility", f"{best_port['Volatilities'] * 100:.2f}%")

        st.divider()

        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("Efficient Frontier")
            fig = px.scatter(
                portfolio,
                x="Volatilities",
                y="Returns",
                color=sharpe,
                color_continuous_scale="Viridis",
                title="Monte Carlo Simulation"
            )
            fig.add_trace(go.Scatter(
                x=[best_port["Volatilities"]],
                y=[best_port["Returns"]],
                mode='markers',
                marker=dict(color='red', size=20, symbol='star')
            ))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Allocation Comparison")

            comp_data = []
            for s in valid_stocks:
                comp_data.append({"Asset": s, "Type": "Current", "Weight": user_weights.get(s, 0)})
                comp_data.append({"Asset": s, "Type": "Optimal", "Weight": best_port[s] * 100})

            fig2 = px.bar(
                pd.DataFrame(comp_data),
                x="Asset",
                y="Weight",
                color="Type",
                barmode="group",
                color_discrete_map={"Current": "#94a3b8", "Optimal": "#22c55e"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Rebalancing Action Plan")
        st.dataframe(pd.DataFrame(rebal), use_container_width=True)