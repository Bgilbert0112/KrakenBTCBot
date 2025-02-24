import os
import time
import pandas as pd
import talib as ta
from dotenv import load_dotenv
from krakenex import API
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load API keys from .env (optional for public OHLC, but included for consistency)
load_dotenv()
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Initialize Kraken API
kraken = API(key=API_KEY, secret=API_SECRET)

# Trading parameters
PAIR = "XXBTZUSD"
INTERVAL = 60  # 1-hour candles
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_PERIOD = 12
WINDOW = 12  # Increased lookback window for local highs/lows (12 hours)


def fetch_historical_data(start_time, end_time):
    """Fetch 1 month of OHLC data from Kraken (720 candles max)."""
    print("Fetching 1 month of historical OHLC data from Kraken...")
    df_list = []
    current_since = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    try:
        while current_since < end_timestamp:
            try:
                resp = kraken.query_public("OHLC", {"pair": PAIR, "interval": INTERVAL, "since": current_since})
                if "error" in resp and resp["error"]:
                    print(f"API error: {resp['error']}")
                    break
                ohlc = resp["result"].get(PAIR, [])
                if not ohlc:
                    print(f"No data for pair {PAIR} at timestamp {current_since}")
                    break
                df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df["close"] = df["close"].astype(float)
                df_list.append(df[["time", "close", "high", "low", "volume"]])  # Include high, low, volume

                # Get the last timestamp from the response
                new_last = int(resp["result"]["last"])
                print(f"Retrieved data up to {df['time'].iloc[-1]} ({len(df)} candles), last timestamp: {new_last}")

                # Check for progress and handle duplicates
                if new_last <= current_since:
                    print(f"No progress in timestamp (stuck at {new_last}). Stopping fetch.")
                    break
                current_since = new_last + 1  # Advance to next timestamp

                # Cap at ~720 candles (1 month at 1-hour intervals)
                total_candles = sum(len(df) for df in df_list)
                if total_candles >= 720:
                    print(f"Reached maximum candles (720) for 1 month. Stopping fetch.")
                    break

                time.sleep(1)  # Respect Kraken rate limits (1 request/sec public)
            except Exception as e:
                print(f"Fetch error: {e}")
                break
    except KeyboardInterrupt:
        print("\nFetch interrupted by user. Saving partial data...")
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time")
            combined_df.to_csv("partial_historical_data.csv", index=False)
            print("Partial data saved to partial_historical_data.csv")
        return pd.DataFrame(columns=["time", "close", "high", "low", "volume"])

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Remove duplicates and sort by time
        combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time")
        return combined_df
    return pd.DataFrame(columns=["time", "close", "high", "low", "volume"])


def calculate_indicators(df):
    """Calculate technical indicators using talib."""
    df["rsi"] = ta.RSI(df["close"], timeperiod=RSI_PERIOD)
    df["macd"], df["macd_signal"], _ = ta.MACD(df["close"], fastperiod=MACD_FAST, slowperiod=MACD_SLOW,
                                               signalperiod=MACD_SIGNAL)
    df["ema"] = ta.EMA(df["close"], timeperiod=EMA_PERIOD)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100  # Bollinger Band % width
    df["stoch_k"], df["stoch_d"] = ta.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3,
                                            slowd_period=3)
    df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)  # Volume moving average
    return df


def label_signals(df, window=WINDOW):
    """Label candles as 'buy' (local low), 'sell' (local high), or 'hold' (neither)."""
    signals = []
    for i in range(len(df)):
        if i < window or i >= len(df) - window:
            signals.append("hold")
            continue

        window_closes = df.iloc[i - window:i + window]["close"]
        is_low = df.iloc[i]["close"] <= window_closes.min()
        is_high = df.iloc[i]["close"] >= window_closes.max()

        if is_low:
            signals.append("buy")
        elif is_high:
            signals.append("sell")
        else:
            signals.append("hold")
    df["signal"] = signals
    return df


def prepare_features(df):
    """Prepare features for ML model."""
    features = df[
        ["close", "rsi", "macd", "macd_signal", "ema", "bb_width", "stoch_k", "stoch_d", "volume", "volume_ma"]]
    # Add lagged features
    for lag in range(1, 4):  # Lags of 1, 2, 3 hours
        features[f"close_lag{lag}"] = df["close"].shift(lag)
        features[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
        features[f"macd_lag{lag}"] = df["macd"].shift(lag)
    features = features.dropna()
    return features


def train_ml_model(df):
    """Train a Random Forest Classifier to predict buy/sell signals, filtering RSI 41–59."""
    # Label signals
    labeled_df = label_signals(df.copy())

    # Prepare features and labels
    features = prepare_features(labeled_df)
    labels = labeled_df.loc[features.index, "signal"].map({"buy": 0, "sell": 1, "hold": 2})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Predict on full dataset
    X_scaled = scaler.transform(features)
    predictions = rf.predict(X_scaled)
    predicted_signals = pd.Series(predictions, index=features.index).map({0: "buy", 1: "sell", 2: "hold"})

    # Filter signals based on RSI (block 41–59)
    final_signals = []
    for idx, signal in predicted_signals.items():
        rsi = labeled_df.loc[idx, "rsi"]
        if signal in ["buy", "sell"]:
            if 41 <= rsi <= 59:
                final_signals.append("hold")
            else:
                final_signals.append(signal)
        else:
            final_signals.append("hold")

    labeled_df.loc[features.index, "signal"] = final_signals

    return labeled_df, rf, scaler


def plot_signals(price_df):
    """Plot Bitcoin price with ML-predicted buy/sell signals over 1 month, filtered by RSI."""
    plt.figure(figsize=(12, 6))
    plt.plot(price_df["time"], price_df["close"], label="BTC/USD Price (XXBTZUSD)", color="#1f77b4")

    # Plot buy signals (green up arrows)
    buy_signals = price_df[price_df["signal"] == "buy"]
    plt.scatter(buy_signals["time"], buy_signals["close"], marker="^", color="#00FF00", label="Buy Signal", s=100)

    # Plot sell signals (red down arrows)
    sell_signals = price_df[price_df["signal"] == "sell"]
    plt.scatter(sell_signals["time"], sell_signals["close"], marker="v", color="#FF0000", label="Sell Signal", s=100)

    plt.title("Bitcoin Price (XXBTZUSD) and Trading Signals - Past 1 Month (ML with RSI Filter)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Set date range (1 month back from today, Feb 23, 2025, 4:44 PM PST = Feb 24, 2025, 00:44 UTC)
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=30)  # 1 month = 30 days

    # Fetch historical data
    price_data = fetch_historical_data(start_time, end_time)
    if price_data.empty:
        print("No historical data retrieved. Check API or internet connection.")
        return

    # Calculate indicators
    price_data = calculate_indicators(price_data)

    # Train ML model and generate signals with RSI filter
    price_data, _, _ = train_ml_model(price_data)

    # Plot the results
    plot_signals(price_data)

    # Print summary of signals
    buys = len(price_data[price_data["signal"] == "buy"])
    sells = len(price_data[price_data["signal"] == "sell"])
    print(f"Total Buy Signals: {buys}")
    print(f"Total Sell Signals: {sells}")
    print(f"Price Data Sample:\n{price_data.tail()}")


if __name__ == "__main__":
    main()