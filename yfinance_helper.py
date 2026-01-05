"""
YFinance Helper Module
Provides robust wrapper functions for yfinance API with proper error handling
and holiday/market closure handling for Indian stocks
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def fetch_stock_data_robust(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    progress: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using yfinance with proper error handling
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TCS.NS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1wk', '1mo')
        progress: Show progress bar
        
    Returns:
        DataFrame with OHLCV data, or None on error
    """
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Download data - don't use 'quiet' parameter as it's not supported in newer versions
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=progress
        )
        
        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
        
        # Handle MultiIndex columns (when downloading multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, level=1, axis=1)
        
        # Reset index to have date as column
        data.reset_index(inplace=True)
        
        # Normalize column names
        data.columns = [col.lower() if isinstance(col, str) else col for col in data.columns]
        
        logger.info(f"Successfully fetched {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_latest_price(
    ticker: str,
    max_lookback_days: int = 10
) -> Optional[Dict]:
    """
    Fetch latest available price for a stock
    If market is closed on requested date, checks prior trading days
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TCS.NS')
        max_lookback_days: Maximum days to look back (for holidays/weekends)
        
    Returns:
        Dictionary with OHLCV data, or None if not available
    """
    try:
        end_date = datetime.now().date()
        
        # Try fetching for today and prior days (in case of holiday/weekend)
        for days_back in range(max_lookback_days):
            check_date = end_date - timedelta(days=days_back)
            
            try:
                data = yf.download(
                    ticker,
                    start=check_date,
                    end=check_date + timedelta(days=1),
                    progress=False
                )
                
                if len(data) > 0:
                    row = data.iloc[0]
                    if days_back > 0:
                        logger.info(
                            f"Data for {ticker}: Found on {check_date} "
                            f"(market may have been closed on {end_date})"
                        )
                    
                    return {
                        'ticker': ticker,
                        'date': check_date,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume'])
                    }
            except Exception as inner_e:
                logger.debug(f"No data for {check_date}: {inner_e}")
                continue
        
        logger.warning(
            f"No data found for {ticker} on {end_date} "
            f"or prior {max_lookback_days} trading days"
        )
        return None
        
    except Exception as e:
        logger.error(f"Error fetching latest price for {ticker}: {e}")
        return None


def fetch_date_range_with_fallback(
    ticker: str,
    target_date: datetime,
    max_lookback_days: int = 5
) -> Optional[Dict]:
    """
    Fetch OHLCV data for a target date
    If no data on target date (holiday), checks prior trading days
    
    Args:
        ticker: Stock ticker symbol
        target_date: Target date to fetch
        max_lookback_days: Maximum days to look back
        
    Returns:
        Dictionary with OHLCV data and actual_date, or None
    """
    try:
        # Try target date and prior days
        for days_back in range(max_lookback_days):
            check_date = target_date - timedelta(days=days_back)
            
            try:
                data = yf.download(
                    ticker,
                    start=check_date.date(),
                    end=check_date.date() + timedelta(days=1),
                    progress=False
                )
                
                if len(data) > 0:
                    row = data.iloc[0]
                    actual_date = check_date if days_back > 0 else target_date
                    
                    if days_back > 0:
                        logger.info(
                            f"{ticker}: Data for {check_date.date()} "
                            f"(requested {target_date.date()}, market closed)"
                        )
                    
                    return {
                        'ticker': ticker,
                        'requested_date': target_date.date(),
                        'actual_date': actual_date.date(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']),
                        'days_lookback': days_back
                    }
            except Exception as inner_e:
                logger.debug(f"No data for {check_date.date()}: {inner_e}")
                continue
        
        logger.warning(
            f"{ticker}: No data found for {target_date.date()} "
            f"or prior {max_lookback_days} trading days"
        )
        return None
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker} on {target_date}: {e}")
        return None


if __name__ == "__main__":
    # Test the helper functions
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test fetching latest price
    print("\n=== Testing Latest Price Fetch ===")
    result = fetch_latest_price('TCS.NS')
    if result:
        print(f"✅ {result['ticker']}: ₹{result['close']:.2f} on {result['date']}")
    
    # Test historical data
    print("\n=== Testing Historical Data Fetch ===")
    data = fetch_stock_data_robust(
        'TCS.NS',
        '2025-12-01',
        '2025-12-30'
    )
    if data is not None:
        print(f"✅ Fetched {len(data)} records")
        print(data.head())
