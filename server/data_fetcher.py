import akshare as ak
import baostock as bs
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
    
    # 东方财富接口可用性熔断机制
    _em_available = True
    _last_probe_time = 0
    _probe_interval = 3600 # 1小时检查一次健康度

    # BaoStock 会话管理
    _bs_logged_in = False

    def __init__(self):
        pass
    
    @classmethod
    def ensure_bs_login(cls):
        """确保 BaoStock 已登录，避免重复登录开销"""
        if not cls._bs_logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                cls._bs_logged_in = True
        return cls._bs_logged_in

    @classmethod
    def ensure_bs_logout(cls):
        """显式登出 (谨慎使用，仅在长任务结束时调用)"""
        try:
            if cls._bs_logged_in:
                bs.logout()
                cls._bs_logged_in = False
        except:
            pass
    
    @classmethod
    def probe_em_health(cls):
        """
        前置探针：通过获取 000001 的数据测试东方财富接口是否可用。
        """
        current_time = time.time()
        # 如果距离上次检查不足 1 小时且已知不可用，则维持现状
        if not cls._em_available and (current_time - cls._last_probe_time < cls._probe_interval):
            return False
            
        print("[Fetcher] 🏥 Probing EastMoney (EM) interface health with '000001'...")
        cls._last_probe_time = current_time
        
        success = False
        for i in range(3): # 尝试 3 次
            try:
                # 快速请求，不带复杂重试逻辑
                df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", adjust="qfq")
                if not df.empty:
                    success = True
                    break
            except Exception:
                time.sleep(1)
        
        cls._em_available = success
        if not success:
            print("[Fetcher] 🚨 EastMoney probe FAILED. Circuit broken. Switching to Fallback sources.")
        else:
            print("[Fetcher] ✅ EastMoney probe PASSED. Using EM as primary source.")
        return success

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
        try:
            cls._spot_cache = ak.stock_zh_a_spot_em()
            cls._last_spot_time = current_time
        except Exception as e:
            print(f"[Fetcher] Error fetching spot data: {e}. Attempting rich fallback (Sina)...")
            # 尝试一个数据更全的备用接口 (Sina Rich)
            try:
                cls._spot_cache = ak.stock_zh_a_spot_sina()
                cls._last_spot_time = current_time
            except Exception as e2:
                print(f"[Fetcher] Rich fallback failed: {e2}. Trying basic backup...")
                try:
                    cls._spot_cache = ak.stock_zh_a_spot()
                    cls._last_spot_time = current_time
                except Exception:
                    # 如果都失败了，且本地有旧缓存，勉强用一下旧的
                    if os.path.exists(cache_file):
                        with open(cache_file, 'rb') as f:
                            cls._spot_cache = pickle.load(f)
                            return
                    raise e
        
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

    @classmethod
    def get_kline_data(cls, symbol: str, period: str = "daily", start_date: str = None, days: int = 200) -> pd.DataFrame:
        """获取历史 K 线数据 (带重试机制与熔断保护)"""
        max_retries = 3
        retry_delay = 5 
            
        # 💡 统一代码格式：ak.stock_zh_a_hist 只要 6 位数字
        clean_symbol = "".join(filter(str.isdigit, str(symbol)))
                
        if not cls._em_available:
            return cls._fetch_fallback_kline(clean_symbol, start_date, days)
        
        for attempt in range(max_retries):
            try:
                if not start_date:
                    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
                    
                df = ak.stock_zh_a_hist(symbol=clean_symbol, period=period, start_date=start_date, adjust="qfq")
                if df.empty: return pd.DataFrame()
                return df
            except Exception as e:
                error_msg = str(e)
                if 'Connection aborted' in error_msg or 'RemoteDisconnected' in error_msg:
                    if attempt == 0 and not cls.probe_em_health():
                        return cls._fetch_fallback_kline(clean_symbol, start_date, days)
                    time.sleep(retry_delay * (attempt + 1))
                if attempt == max_retries - 1:
                    return cls._fetch_fallback_kline(clean_symbol, start_date, days)
        return pd.DataFrame()
    
    @classmethod
    def _fetch_fallback_kline(cls, symbol, start_date, days=200):
        """内部备选抓取逻辑 (优先 BaoStock, 次选 Sina)"""
        # 1. 尝试使用 BaoStock
        try:
            raw_symbol = "".join(filter(str.isdigit, str(symbol)))
            prefix = "sh" if raw_symbol.startswith("6") else "sz"
            bs_symbol = f"{prefix}.{raw_symbol}"
            
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
            elif "-" not in str(start_date):
                start_date = f"{str(start_date)[:4]}-{str(start_date)[4:6]}-{str(start_date)[6:]}"
            
            cls.ensure_bs_login()
            rs = bs.query_history_k_data_plus(
                bs_symbol, "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date, end_date=datetime.date.today().strftime("%Y-%m-%d"),
                frequency="d", adjustflag="2"
            )
            
            if rs.error_code == '0':
                data_list = []
                while rs.next(): data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    for col in ["open","high","low","close","volume","amount","turn","pctChg"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df.rename(columns={
                        "date":"日期","open":"开盘","high":"最高","low":"最低","close":"收盘",
                        "volume":"成交量","amount":"成交额","turn":"换手率","pctChg":"涨跌幅"
                    })
        except Exception: pass

        # 2. 尝试使用 Sina
        try:
            prefix = "sh" if str(symbol).startswith("6") else ("sz" if str(symbol).startswith(("0", "3")) else "bj")
            df_sina = ak.stock_zh_a_daily(symbol=f"{prefix}{symbol}", start_date=str(start_date).replace("-",""), adjust="qfq")
            if df_sina is not None and not df_sina.empty:
                df_sina = df_sina.rename(columns={"date":"日期","open":"开盘","high":"最高","low":"最低","close":"收盘","volume":"成交量","amount":"成交额","turnover":"换手率"})
                if "收盘" in df_sina.columns:
                    df_sina['涨跌幅'] = df_sina['收盘'].pct_change() * 100
                if "换手率" in df_sina.columns: df_sina['换手率'] = df_sina['换手率'] * 100
                return df_sina
        except Exception: pass
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
