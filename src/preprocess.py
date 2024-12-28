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


if __name__ == "__main__":
    tickers = yf.Tickers('AAPL MSFT GOOGL AMZN FB NVDA') # Get multiple tickers
    for ticker in tickers.tickers:
        data = ticker.history(period="max")
        data.to_csv(f"data/{ticker.ticker}.csv")
        print(f"Data saved to data/{ticker.ticker}.csv")
         