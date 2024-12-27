import yfinance as yf
import os

def fetch_stock_data(ticker, start_date, end_date, save_path):
    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        # Save to CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
    else:
        print(f"No data found for ticker {ticker}.")

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    save_path = "data/apple_stock_data.csv"
    fetch_stock_data(ticker, start_date, end_date, save_path)
