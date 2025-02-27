# KrakenBot - Bitcoin Trading Bot with Machine Learning

## Overview
KrakenBot is an automated trading bot for Bitcoin (XXBTZUSD) on the Kraken exchange, using Machine Learning (Random Forest) and technical indicators (RSI, MACD, EMA, Stochastic, Bollinger Bands, Volume) to predict buy/sell signals. It targets major price movements (e.g., $94,800 lows and $96,325.80 highs) while filtering noise and ensuring profitability after Kraken trading fees (~0.26% round-trip). The bot trades with 50% of available balance for safety and uses an RSI filter to block trades in the neutral 41–59 range.

## Features
- Real-time trading with 1-hour OHLC data from Kraken.
- ML-based predictions using a Random Forest Classifier trained on historical data.
- RSI filter to avoid choppy trades (buys < RSI 41, sells > RSI 59).
- 50% balance limit for risk management.
- run data_generation.py to test bot on historical data.
- run Budget_UI.py to view API dashboard.

## Prerequisites
- **Python 3.9–3.11** (recommended: 3.11)
- **Kraken API Key and Secret** (store in `.env` file)
- Linux or Windows system (tested on Ubuntu and Windows 11)

## Installation

### 1. Clone the Repository

bash:

git clone https://github.com/bgilbert0112/KrakenBot.git
cd KrakenBot

### 2. Set Up Virtual Environment
Create and activate a virtual environment:

bash:

python3 -m venv .venv

source .venv/bin/activate  # Linux/Mac

.\venv\Scripts\activate    # Windows

### 3. Install Dependencies
Install required packages from requirements.txt:

bash:

pip install -r requirements.txt
Note: If TA-Lib isn’t available via pip, install it manually:

Linux: Download and install the TA-Lib binary from ta-lib.org, then pip install TA-Lib==0.4.28.
Windows: Use a pre-built wheel or install via conda or binary.

### 4. Configure Kraken API Credentials
Create a .env file in the KrakenBot directory:

bash:

nano .env  # Linux
notepad .env  # Windows

Add your Kraken API keys:

text:

KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

Save and exit.

Usage
Local Testing
Run the bot manually to test:

bash:

.venv/bin/python real_time_trading_ml.py  # Linux
.\venv\Scripts\python real_time_trading_ml.py  # Windows
Monitor output for trades at $94,800 buys and $96,325.80 sells with 50% balance, holding in neutral RSI ranges.

### 5. Contributing:
Fork the repository.
Create a branch for your changes.
Submit a pull request with tests and documentation.

### 6. Support the Developer

Love this program? Want to buy me a coffee to support further development? Your support is greatly appreciated! You can send a donation via Bitcoin to the following wallet address:

Bitcoin Wallet: bc1qwgpr8keenl8zggnt2823wpkjmjky5ga8pzuvkd

Thank you for your support!