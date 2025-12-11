# ML Smart Portfolio Rebalancer
A machine-learning powered portfolio optimization tool that predicts volatility and generates optimal asset allocation using EGARCH, XGBoost, and Monte Carlo simulation.

## Features
- Volatility Prediction: EGARCH + XGBoost hybrid model
- Smart Returns: Blended arithmetic & geometric mean
- Portfolio Optimization: Monte Carlo simulation with Sharpe ratio maximization
- Interactive Dashboard: Streamlit web interface
- Rebalancing: Automated Buy/Sell/Hold recommendations

## Quick Start
Installation
```bash
pip install -r requirements.txt
```
Run the App
```bash
streamlit run streamlit_app.py
```
## How It Works
### Volatility Prediction
- EGARCH(1,1): Captures volatility clustering
- XGBoost: Uses engineered features (returns, volume, lags)
- Output: Next-day annualized volatility

## Optimisation
Monte Carlo simulation → Maximum Sharpe ratio → Optimal weights

## Authors
- Dexun
- Thaddus
- Eron
