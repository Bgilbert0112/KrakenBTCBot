import os
import time
import pandas as pd
import talib as ta
from dotenv import load_dotenv
from krakenex import API
from threading import Thread
import sys
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load API keys from .env
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
WINDOW = 12  # Lookback window for local highs/lows (12 hours)
UPDATE_INTERVAL = 900  # 15 minutes between updates
TRADE_PERCENT = 0.4  # Use 40% of available balance for buffer

# Global state
price_data = pd.DataFrame(columns=["time", "close", "high", "low", "volume"])
model = None
scaler = None


def fetch_historical_data(start_time, end_time):
    """
    Fetch 1 month of OHLC (Open, High, Low, Close) data from Kraken for initial model training.
    Limits to 720 candles (1-hour intervals over 30 days) to avoid rate limits and ensure efficiency.

    Args:
        start_time (datetime): Start date/time for historical data (e.g., 30 days ago).
        end_time (datetime): End date/time for historical data (e.g., now).

    Returns:
        pd.DataFrame: DataFrame with columns ['time', 'close', 'high', 'low', 'volume'] or empty if no data.
    """
    # Inform the user that we're starting to fetch historical OHLC data for the initial ML model
    print("Fetching 1 month of historical OHLC data from Kraken for initial model training...")

    # Initialize an empty list to store DataFrames of OHLC data chunks fetched from Kraken
    df_list = []

    # Convert datetime objects to Unix timestamps (seconds since epoch) for Kraken API
    current_since = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    try:
        # Loop to fetch data in chunks until we reach the end time or hit the candle limit (720)
        while current_since < end_timestamp:
            try:
                # Query Kraken's public OHLC API for the specified pair (XXBTZUSD), interval (1 hour),
                # and starting timestamp, respecting Kraken’s rate limits
                resp = kraken.query_public("OHLC", {"pair": PAIR, "interval": INTERVAL, "since": current_since})

                # Check if the API response contains errors (e.g., rate limit, invalid pair)
                if "error" in resp and resp["error"]:
                    print(f"API error: {resp['error']}")
                    break  # Exit the inner loop if there's an error

                # Extract OHLC data for the specified pair from the response
                ohlc = resp["result"].get(PAIR, [])

                # If no OHLC data is returned for the pair, log an error and stop fetching
                if not ohlc:
                    print(f"No data for pair {PAIR} at timestamp {current_since}")
                    break

                # Convert the OHLC list into a pandas DataFrame with specific column names
                df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])

                # Convert the Unix timestamp (in seconds) to a readable datetime for the 'time' column
                df["time"] = pd.to_datetime(df["time"], unit="s")

                # Ensure the 'close' price is a float for numerical operations
                df["close"] = df["close"].astype(float)

                # Append only the necessary columns ('time', 'close', 'high', 'low', 'volume') to the list
                # This reduces memory usage by excluding less critical data like 'open', 'vwap', 'count'
                df_list.append(df[["time", "close", "high", "low", "volume"]])

                # Get the last timestamp from the response to continue fetching subsequent data
                # Kraken provides 'last' as the timestamp of the last candle fetched
                new_last = int(resp["result"]["last"])

                # Log the progress, showing the latest timestamp and number of candles fetched
                print(f"Retrieved data up to {df['time'].iloc[-1]} ({len(df)} candles), last timestamp: {new_last}")

                # Check if we've hit a loop (no progress in timestamp), indicating we’ve fetched all available data
                if new_last <= current_since:
                    print(f"No progress in timestamp (stuck at {new_last}). Stopping fetch.")
                    break

                # Update the starting timestamp for the next API call to fetch the next batch of data
                current_since = new_last + 1

                # Count total candles across all chunks to limit to ~720 (1 month at 1-hour intervals)
                total_candles = sum(len(df) for df in df_list)
                if total_candles >= 720:
                    print(f"Reached maximum candles (720) for 1 month. Stopping fetch.")
                    break

                # Pause for 1 second to respect Kraken’s public API rate limits (1 request/sec max)
                time.sleep(1)

            except Exception as e:
                # Handle any errors during the API call (e.g., network issues, rate limits)
                print(f"Fetch error: {e}")
                break  # Exit the inner loop if an error occurs

    except KeyboardInterrupt:
        # Handle manual interruption (e.g., Ctrl+C) by saving partial data to a CSV file
        print("\nFetch interrupted by user. Saving partial data...")
        if df_list:
            # Combine all DataFrames in df_list into one, reset index, and remove duplicates by 'time'
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time")

            # Save the partial data to a CSV file for recovery or analysis
            combined_df.to_csv("partial_historical_data.csv", index=False)
            print("Partial data saved to partial_historical_data.csv")

        # Return an empty DataFrame to indicate interruption
        return pd.DataFrame(columns=["time", "close", "high", "low", "volume"])

    # If data was successfully fetched, combine all chunks, remove duplicates, and sort by time
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Remove any duplicate entries based on timestamp and sort chronologically
        combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time")
        return combined_df

    # If no data was fetched (e.g., API errors or no data available), return an empty DataFrame
    return pd.DataFrame(columns=["time", "close", "high", "low", "volume"])


def fetch_prices():
    """Fetch OHLC data from Kraken every 5 minutes with detailed logging."""
    global price_data
    print("🚀 Fetch thread started!")
    retries = 0
    max_retries = 3
    while True:
        try:
            resp = kraken.query_public("OHLC", {"pair": PAIR, "interval": INTERVAL})
            if "error" in resp and resp["error"]:
                raise Exception(f"API error: {resp['error']}")
            ohlc = resp["result"].get(PAIR, None)
            if not ohlc:
                raise Exception(f"No data for pair {PAIR} in response: {resp}")
            df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df["close"] = df["close"].astype(float)
            price_data = df[["time", "close", "high", "low", "volume"]].tail(50)  # Keep last 50 candles
            print(f"✅ Fetched {len(price_data)} candles at {time.ctime()}")
            retries = 0
        except Exception as e:
            retries += 1
            print(f"Fetch error ({retries}/{max_retries}): {str(e)}")
            if retries >= max_retries:
                print("Max retries reached. Pausing fetch for 15 minutes.")
                time.sleep(UPDATE_INTERVAL)
                retries = 0
            time.sleep(10 * retries)
        time.sleep(UPDATE_INTERVAL)


def get_balance():
    """Fetch USD and BTC balances with retry logic."""
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            resp = kraken.query_private("Balance")
            if "error" in resp and resp["error"]:
                raise Exception(f"API error: {resp['error']}")
            usd = float(resp["result"].get("ZUSD", 0))
            btc = float(resp["result"].get("XXBT", 0))
            return usd, btc
        except Exception as e:
            retries += 1
            print(f"Balance fetch error ({retries}/{max_retries}): {e}")
            time.sleep(10 * retries)
    print("Failed to fetch balance after retries.")
    return 0, 0


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
    return signals


def prepare_features(df):
    """Prepare features for ML model, avoiding SettingWithCopyWarning."""
    # Create a new DataFrame instead of modifying a slice
    features = pd.DataFrame(index=df.index)
    features["close"] = df["close"]
    features["rsi"] = df["rsi"]
    features["macd"] = df["macd"]
    features["macd_signal"] = df["macd_signal"]
    features["ema"] = df["ema"]
    features["bb_width"] = df["bb_width"]
    features["stoch_k"] = df["stoch_k"]
    features["stoch_d"] = df["stoch_d"]
    features["volume"] = df["volume"]
    features["volume_ma"] = df["volume_ma"]

    # Add lagged features using .loc to avoid warnings
    for lag in range(1, 4):  # Lags of 1, 2, 3 hours
        features.loc[:, f"close_lag{lag}"] = df["close"].shift(lag)
        features.loc[:, f"rsi_lag{lag}"] = df["rsi"].shift(lag)
        features.loc[:, f"macd_lag{lag}"] = df["macd"].shift(lag)

    return features.dropna()


def train_initial_model():
    """Train initial ML model on historical data (1 month back)."""
    global model, scaler
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=30)

    historical_data = fetch_historical_data(start_time, end_time)
    if historical_data.empty:
        print("No historical data for initial training. Using empty model.")
        return None, None

    historical_data = calculate_indicators(historical_data)
    labeled_df = pd.DataFrame()
    labeled_df["signal"] = label_signals(historical_data)
    labeled_df = pd.concat([historical_data, labeled_df], axis=1)

    features = prepare_features(labeled_df)
    labels = labeled_df.loc[features.index, "signal"].map({"buy": 0, "sell": 1, "hold": 2})

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    model = rf
    return rf, scaler


def predict_signal(df):
    """Predict buy/sell/hold signal using trained ML model, filtered by RSI 41–59."""
    if model is None or scaler is None:
        print("Model not trained. Returning 'hold'.")
        return "hold"

    # Prepare features
    features = prepare_features(df)
    if features.empty:
        return "hold"

    # Scale features
    X_scaled = scaler.transform(features)

    # Predict (returns NumPy array)
    prediction = model.predict(X_scaled)[-1]  # Get the last prediction

    # Map prediction to signal
    signal = {0: "buy", 1: "sell", 2: "hold"}[prediction]

    # Filter by RSI (block 41–59)
    rsi = df.iloc[-1]["rsi"]
    if signal in ["buy", "sell"]:
        if 41 <= rsi <= 59:
            return "hold"
    return signal


def execute_trade(signal, price, usd_balance, btc_balance):
    """Execute a market order on Kraken using 40% of available balance."""
    if signal == "hold":
        print("⏸️ Holding position.")
        return

    trade_amount = min(btc_balance * TRADE_PERCENT, 0.001) if signal == "sell" else (
                                                                                                usd_balance * TRADE_PERCENT) / price
    if trade_amount < 0.0001:
        print(f"🚨 Insufficient funds to {signal.upper()} ({trade_amount:.6f} BTC too small).")
        return

    side = "buy" if signal == "buy" else "sell"
    try:
        order = kraken.query_private("AddOrder", {
            "pair": PAIR,
            "type": side,
            "ordertype": "market",
            "volume": f"{trade_amount:.6f}"
        })
        if "error" in order and order["error"]:
            print(f"❌ Trade error: {order['error']}")
        else:
            print(f"🎯 {side.upper()} executed: {trade_amount:.6f} BTC at ${price:.2f}")
    except Exception as e:
        print(f"❌ Execution error: {e}")


def main():
    """Main trading loop with ML model."""
    global model, scaler
    print("Starting live spot trading bot with ML model...")

    # Train initial model
    model, scaler = train_initial_model()
    if model is None:
        print("Warning: Starting without a trained model. Initial trades may be limited.")

    # Start fetch thread
    fetch_thread = Thread(target=fetch_prices)
    fetch_thread.daemon = True
    fetch_thread.start()
    time.sleep(5)  # Give fetch thread a moment to kick off

    while True:
        if not price_data.empty and len(price_data) >= WINDOW + 1:
            df = price_data.copy()
            df = calculate_indicators(df)

            usd_balance, btc_balance = get_balance()
            signal = predict_signal(df)

            status = ("Flashing BUY signal" if signal == "buy" else
                      "Flashing SELL signal" if signal == "sell" else
                      "Holding position")
            print(f"📊 Time: {df['time'].iloc[-1]}, Price: ${df['close'].iloc[-1]:.2f}, "
                  f"RSI: {df['rsi'].iloc[-1]:.2f}, MACD: {df['macd'].iloc[-1]:.6f}, "
                  f"EMA: {df['ema'].iloc[-1]:.2f}, BB Width: {df['bb_width'].iloc[-1]:.2f}, "
                  f"Stoch K: {df['stoch_k'].iloc[-1]:.2f}, Stoch D: {df['stoch_d'].iloc[-1]:.2f}, "
                  f"Volume: {float(df['volume'].iloc[-1]):.2f}, {status}")

            execute_trade(signal, df["close"].iloc[-1], usd_balance, btc_balance)
        else:
            print(f"⏳ Waiting for sufficient data... (Rows: {len(price_data)})")

        sys.stdout.flush()
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()