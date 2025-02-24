import os
import time
import tkinter as tk
from dotenv import load_dotenv
from krakenex import API
import random

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Initialize Kraken API
kraken = API(key=API_KEY, secret=API_SECRET)

# Motivational financial quotes
QUOTES = [
    "Success is doing ordinary things extraordinarily well.",
    "The stock market rewards patience, not impatience.",
    "True profitability lies beyond comfort zones.",
    "Measure success by your discipline, not the market's whims.",
    "Uncertainty stems from ignorance, not risk.",
    "Invest with consistency, not speculation.",
    "Knowledge yields the highest returns.",
    "Wealth is enjoyed, not merely possessed.",
    "A trader's triumph is in strategy, not just profit.",
    "Patience is your ally; haste, your foe.",
    "Fortune favors the disciplined investor, not the reckless gambler.",
    "The path to wealth is paved with patience and prudent decisions.",
    "In the world of investing, consistency trumps brilliance.",
    "Every great fortune begins with a single, wise investment.",
    "Master your emotions, and the market will follow your lead.",
    "Success in finance is measured by long-term growth, not short-term gains.",
    "A diversified portfolio is the armor of the prudent investor.",
    "The art of investing lies in managing risk, not chasing returns.",
    "Trust in your strategy, for the market rewards conviction.",
    "Wealth is built through steady accumulation, not fleeting speculation."
]

# Fetch current BTC/USD price
def get_btc_price():
    try:
        resp = kraken.query_public("Ticker", {"pair": "XXBTZUSD"})
        if "error" in resp and resp["error"]:
            return None
        return float(resp["result"]["XXBTZUSD"]["c"][0])  # Last close price
    except Exception:
        return None

# Fetch account balances
def get_balances():
    try:
        resp = kraken.query_private("Balance")
        if "error" in resp and resp["error"]:
            return "Error: " + ", ".join(resp["error"])
        usd = float(resp["result"].get("ZUSD", 0))
        btc = float(resp["result"].get("XXBT", 0))
        btc_price = get_btc_price()
        btc_usd = btc * btc_price if btc_price else "N/A"
        return ("USD: ", f"${usd:.2f}", "\nBTC: ", f"{btc:.6f}", " (~$", f"{btc_usd:.2f}" if btc_usd != "N/A" else "N/A", ")")
    except Exception as e:
        return ("Balance Error: ", str(e))

# Fetch last 5 trades
def get_last_trades():
    try:
        resp = kraken.query_private("TradesHistory", {"ofs": 0, "count": 5})
        if "error" in resp and resp["error"]:
            return [("Error: ", ", ".join(resp["error"]))]
        trades = resp["result"]["trades"]
        trade_list = []
        for trade_id, trade in list(trades.items())[:5]:
            pair = trade["pair"]
            trade_type = trade["type"]
            price = float(trade["price"])
            vol = float(trade["vol"])
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(trade["time"])))
            trade_list.append((f"{timestamp} - ", trade_type.upper(), f" {vol:.6f} {pair} @ $", f"{price:.2f}"))
        return trade_list if trade_list else [("No trades yet.",)]
    except Exception as e:
        return [("Trades Error: ", str(e))]

# UI Setup
root = tk.Tk()
root.title("KrakenBot Executive Dashboard")
root.geometry("1200x800")
root.configure(bg="#1e1e1e")

# Balance Frame
balance_frame = tk.Frame(root, bg="#4A4A4A", bd=3, relief="raised")
balance_frame.pack(pady=40, padx=40, fill="x")

balance_label = tk.Label(balance_frame, text="Hello!\nHere are your Kraken account balances:",
                         font=("Arial", 28, "bold"), fg="#FFFFFF", bg="#4A4A4A")
balance_label.pack(pady=15)

quote_label = tk.Label(balance_frame, text=random.choice(QUOTES), font=("Arial", 18, "italic"),
                       fg="#FFD700", bg="#4A4A4A", wraplength=1000, justify="center")
quote_label.pack(pady=15)

balance_text = tk.Label(balance_frame, font=("Arial", 20), bg="#4A4A4A", justify="left")
balance_text.pack(pady=20)

# Trades Frame
trades_frame = tk.Frame(root, bg="#4A4A4A", bd=3, relief="raised")
trades_frame.pack(pady=40, padx=40, fill="x")

trades_label = tk.Label(trades_frame, text="Recent Trading Activity", font=("Arial", 28, "bold"),
                        fg="#FFFFFF", bg="#4A4A4A")
trades_label.pack(pady=15)

trades_text = tk.Label(trades_frame, font=("Arial", 20), bg="#4A4A4A", justify="left")
trades_text.pack(pady=20)

# Update function
def update_ui():
    # Update balances with green amounts
    balance_data = get_balances()
    balance_text.config(text="")
    for part in balance_data:
        if isinstance(part, str) and any(c.isdigit() or c in "$.~" for c in part):
            balance_text.config(text=balance_text.cget("text") + part, fg="#00FF00")
        else:
            balance_text.config(text=balance_text.cget("text") + part, fg="#FFFFFF")

    # Update trades with colored trade types and green amounts
    trades_data = get_last_trades()
    trades_text.config(text="")
    for line_parts in trades_data:
        for i, part in enumerate(line_parts):
            if i == 1 and part == "BUY":
                trades_text.config(text=trades_text.cget("text") + part, fg="#FFA500")  # Orange for BUY
            elif i == 1 and part == "SELL":
                trades_text.config(text=trades_text.cget("text") + part, fg="#0000FF")  # Blue for SELL
            elif isinstance(part, str) and any(c.isdigit() or c in "$.~" for c in part):
                trades_text.config(text=trades_text.cget("text") + part, fg="#00FF00")  # Green for amounts
            else:
                trades_text.config(text=trades_text.cget("text") + part, fg="#FFFFFF")  # White for rest
        trades_text.config(text=trades_text.cget("text") + "\n")

    quote_label.config(text=random.choice(QUOTES))
    root.after(60000, update_ui)

# Start the update loop
update_ui()

# Run the UI
root.mainloop()