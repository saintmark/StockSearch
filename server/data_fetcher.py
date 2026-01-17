import akshare as ak
import pandas as pd
from typing import List, Dict, Optional
import datetime
import time
import pickle
import os

class StockDataFetcher:
    """A 股行情与财务数据获取类 (封装 AkShare)"""
    # 增加实时行情缓存，避免频繁抓取全市场 5000+ 股票数据
    _spot_cache = None
    _last_spot_time = 0
    _cache_duration = 43200 # 缓存 12 小时 (盘后锁定，规避重复抓取)

    def __init__(self):
        pass
    
    @classmethod
    def _load_or_fetch_spot_cache(cls):
        """Helper to load spot cache from memory/disk or fetch from network."""
        current_time = time.time()
        today_str = datetime.date.today().strftime("%Y%m%d")
        cache_file = os.path.join(os.path.dirname(__file__), f"market_spot_{today_str}.pkl")

        # 1. 检查内存缓存是否有效
        if cls._spot_cache is not None and (current_time - cls._last_spot_time < cls._cache_duration):
            return

        # 2. 尝试从本地磁盘读取 (存活 1 天)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Check if the file-based cache is still fresh enough
                    file_mod_time = os.path.getmtime(cache_file)
                    if (current_time - file_mod_time) < cls._cache_duration:
                        cls._spot_cache = cached_data
                        cls._last_spot_time = file_mod_time
                        return
            except Exception as e:
                print(f"Error loading spot cache from file: {e}. Fetching new data.")
                # If loading fails, proceed to fetch new data

        # 3. 实在没有或过期，发起网络请求
        print("[Fetcher] Spot cache expired or not found. Fetching fresh data...")
        cls._spot_cache = ak.stock_zh_a_spot_em()
        cls._last_spot_time = current_time
        
        # 静默保存到本地
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cls._spot_cache, f)
        except Exception as e:
            print(f"Error saving spot cache to file: {e}")

    @classmethod
    def get_active_stocks(cls, top_n: int = 50) -> List[str]:
        """获取全市场成交额最活跃的 Top N 股票 (自动寻迹的优选池)"""
        try:
            # 确保缓存存在
            cls._load_or_fetch_spot_cache()
            
            if cls._spot_cache is None or cls._spot_cache.empty:
                return []
            
            # 按照成交额排序 (由高到低)
            active_df = cls._spot_cache.sort_values(by='成交额', ascending=False).head(top_n)
            return active_df['代码'].tolist()
        except Exception as e:
            print(f"Error getting active stocks: {e}")
            return []

    @staticmethod
    def get_all_stocks() -> pd.DataFrame:
        """获取全市场 A 股基本信息"""
        try:
            stock_info_a_code_name_df = ak.stock_info_a_code_name()
            return stock_info_a_code_name_df
        except Exception as e:
            print(f"Error fetching stock list: {e}")
            return pd.DataFrame()

    @classmethod
    def get_realtime_quotes(cls, symbols: List[str] = []) -> pd.DataFrame:
        """获取个股或全市场的实时行情 (带三级持久化缓存)"""
        try:
            cls._load_or_fetch_spot_cache()
            
            if cls._spot_cache is None or cls._spot_cache.empty:
                return pd.DataFrame()
            
            if not symbols:
                return cls._spot_cache
            
            return cls._spot_cache[cls._spot_cache['代码'].isin(symbols)]
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_kline_data(symbol: str, period: str = "daily", start_date: str = None, days: int = 200) -> pd.DataFrame:
        """获取历史 K 线数据"""
        try:
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
            
            # Print for debugging
            # print(f"[Fetcher] Fetching K-line for {symbol} from {start_date}")
            
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, adjust="qfq")
            if df.empty:
                print(f"[Fetcher] Warning: K-line data for {symbol} is empty.")
            return df
        except Exception as e:
            print(f"[Fetcher] Error fetching K-line data for {symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def get_company_finance(cls, symbol: str) -> List:
        """获取公司核心财务指标 (ROE, PE, PB, 营收, 净利润等)"""
        try:
            # print(f"[Fetcher] Fetching Finance (Latest) for {symbol}")
            
            # 1. 获取财报摘要 (年报数据相对静态，暂不强制缓存，但其耗时较短)
            latest_report = {}
            try:
                abs_df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="主要指标")
                if abs_df is not None and not abs_df.empty and len(abs_df) > 0:
                    latest_report = abs_df.tail(1).iloc[0].to_dict()
            except Exception as e:
                # 某些股票可能没有财务数据，这是正常的
                # print(f"[Fetcher] Warning: Could not fetch financial abstract for {symbol}: {e}")
                latest_report = {}

            # 2. 获取实时估值 (PE/PB) - 使用缓存的全场行情
            valuation = {}
            try:
                current_time = time.time()
                if cls._spot_cache is None or (current_time - cls._last_spot_time > cls._cache_duration):
                    # print(f"[Fetcher] Cache expired. Fetching fresh spot for valuation...")
                    cls._spot_cache = ak.stock_zh_a_spot_em()
                    cls._last_spot_time = current_time
                
                if cls._spot_cache is not None and not cls._spot_cache.empty:
                    target = cls._spot_cache[cls._spot_cache['代码'] == symbol]
                    if not target.empty:
                        info = target.iloc[0].to_dict()
                        valuation = {
                            "市盈率": info.get('市盈率-动态', '--'),
                            "市净率": info.get('市净率', '--')
                        }
            except Exception as e:
                # 估值数据获取失败，使用默认值
                # print(f"[Fetcher] Warning: Could not fetch valuation for {symbol}: {e}")
                valuation = {}

            # 3. 数据聚合
            final_data = {
                "净资产收益率": latest_report.get('净资产收益率', '--'),
                "市盈率": valuation.get('市盈率', '--'),
                "市净率": valuation.get('市净率', '--'),
                "营业收入": latest_report.get('营业总收入', '--'),
                "净利润": latest_report.get('净利润', '--'),
                "销售毛利率": latest_report.get('销售毛利率', '--'),
                "报告期": latest_report.get('报告期', '--')
            }
            
            return [final_data]
        except Exception as e:
            # 静默处理错误，某些股票可能没有财务数据，这是正常的
            # print(f"[Fetcher] Error fetching consolidated finance for {symbol}: {e}")
            return []

    @staticmethod
    def get_stock_info(symbol: str) -> Dict:
        """获取个股基本信息 (名称, 行业, 上市时间等)"""
        try:
            # print(f"[Fetcher] Fetching Individual Info for {symbol}")
            df = ak.stock_individual_info_em(symbol=symbol)
            if df.empty:
                return {}
            # 将 DataFrame (item, value) 结构转换为字典
            return dict(zip(df['item'], df['value']))
        except Exception as e:
            print(f"[Fetcher] Error fetching individual info for {symbol}: {e}")
            return {}
    
    @staticmethod
    def get_market_index(symbol: str = "sh000300", days: int = 200) -> pd.DataFrame:
        """
        获取市场指数数据（用于计算相对收益）
        
        Args:
            symbol: 指数代码，默认沪深300 (sh000300)
            days: 获取天数
            
        Returns:
            包含日期和收盘价的DataFrame
        """
        try:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df.empty:
                return pd.DataFrame()
            # 筛选日期范围
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] >= pd.to_datetime(start_date)]
            return df
        except Exception as e:
            print(f"[Fetcher] Error fetching market index {symbol}: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # 简单的本地测试
    fetcher = StockDataFetcher()
    print("Testing stock list fetch...")
    print(fetcher.get_all_stocks().head())
