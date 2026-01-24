import pandas as pd
import datetime
import time
import random
from data_fetcher import StockDataFetcher
from database import DatabaseManager

class OneNightStrategy:
    def __init__(self, fetcher: StockDataFetcher, db: DatabaseManager):
        self.fetcher = fetcher
        self.db = db
        # 交易费率设置
        self.commission_rate = 0.0003  # 万三佣金
        self.stamp_duty_rate = 0.001   # 千一印花税 (卖出收)
        self.min_commission = 5.0      # 最低佣金5元

    def calculate_fees(self, amount: float, is_buy: bool) -> float:
        """计算手续费"""
        commission = max(self.min_commission, amount * self.commission_rate)
        stamp_duty = amount * self.stamp_duty_rate if not is_buy else 0
        return commission + stamp_duty

    def check_limit_up_history(self, symbol: str, lookback_days: int = 20, cache_only: bool = True) -> bool:
        """
        检查过去 N 天内是否有过涨停
        涨停定义：日涨幅 > 9.5% (简单判定，涵盖主板10%和科创/创业20%)
        
        Args:
            cache_only: 如果为 True，则只使用缓存数据，不发起网络请求（推荐）
        """
        try:
            # 从数据库缓存读取
            df = self.db.get_cached_kline(symbol, max_age_hours=48)  # 2天内的缓存都可用
            
            if df is None or df.empty:
                if cache_only:
                    # 强制缓存模式：没有缓存就直接跳过，不发起网络请求
                    return False
                else:
                    # 允许网络请求模式（仅在特殊情况下使用）
                    delay = 1.0 + random.random() * 2.0
                    print(f"[Strategy] Cache miss for {symbol}, fetching with {delay:.1f}s delay...")
                    time.sleep(delay)
                    
                    df = self.fetcher.get_kline_data(symbol, days=lookback_days + 10)
                    
                    if df is not None and not df.empty:
                        self.db.save_kline(symbol, df)
                    else:
                        return False
            
            # 取最近 N 天 (切片)
            df = df.tail(lookback_days)
            
            # 检查是否有涨幅 > 9.5%
            has_limit_up = (df['涨跌幅'] > 9.5).any()
            return has_limit_up
        except Exception as e:
            print(f"[Strategy] Error checking limit up for {symbol}: {e}")
            return False

    def scan_market(self, progress_callback=None) -> list:
        """
        全市场扫描：应用 6 大过滤条件
        返回符合条件的候选股列表 (包含完整信息)
        """
        print(f"[OneNight] Starting full market scan at {datetime.datetime.now()}...")
        
        # 1. 获取全市场实时行情
        df = self.fetcher.get_realtime_quotes()
        if df.empty:
            print("[OneNight] Error: Failed to fetch market quotes.")
            return []
        
        # 预处理数值列
        numeric_cols = ['涨跌幅', '量比', '总市值', '换手率', '最新价', '成交量', '成交额']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. 基础筛选 (Vectorized filtering for speed)
        # 计算均价 (注意: 成交量单位通常是手, 需 * 100; 成交额是元)
        df['avg_price'] = df['成交额'] / (df['成交量'] * 100)
        
        mask = (
            (df['涨跌幅'] >= 3.0) & (df['涨跌幅'] <= 5.0) &
            (df['量比'] > 1.0) &
            (df['总市值'] <= 200 * 100000000) &
            (df['换手率'] >= 5.0) & (df['换手率'] <= 10.0) &
            (df['最新价'] > df['avg_price'])
        )
        
        candidates_df = df[mask].copy()
        initial_count = len(candidates_df)
        print(f"[OneNight] {initial_count} stocks passed initial basic filters.")
        
        if candidates_df.empty:
            return []

        # 3. 进阶筛选: 20天内有过涨停
        # 按量比降序排列
        candidates_df = candidates_df.sort_values(by='量比', ascending=False)
        potential_stocks = candidates_df['代码'].tolist()
        
        final_candidates = []
        
        # 💡 使用缓存后速度极快，可以检查更多股票
        # 如果缓存已建立，整个过程只需 10-30 秒
        max_check = min(100, len(potential_stocks)) 
        check_count = 0
        
        print(f"[OneNight] Checking limit-up history for top {max_check} candidates (cache-only mode)...")
        
        for symbol in potential_stocks:
            if check_count >= max_check: 
                break
                
            if progress_callback:
                progress_callback(check_count, max_check)
                
            if self.check_limit_up_history(symbol):
                # 获取该股完整信息
                row = candidates_df[candidates_df['代码'] == symbol].iloc[0]
                
                # 构造符合 recommendation 格式的字典
                rec = {
                    'symbol': symbol,
                    'name': row['名称'],
                    'price': float(row['最新价']),
                    'change': float(row['涨跌幅']),
                    'turnover': float(row['换手率']),
                    'industry': row.get('行业', '未知'), # 实时行情可能包含行业
                    'score': float(row['量比']), # 使用量比作为分数
                    'action': 'BUY',
                    'advice': f"一夜持股严选：量比 {row['量比']:.2f}，换手 {row['换手率']:.2f}%",
                    'reasons': [
                        "涨跌幅 3%-5%", "量比 > 1", "换手率 5%-10%", 
                        "市值 <= 200亿", "股价 > 分时均线", "20日内有涨停"
                    ]
                }
                final_candidates.append(rec)
            
            check_count += 1 
            
        print(f"[OneNight] Full scan complete. Found {len(final_candidates)} candidates.")
        return final_candidates

    def daily_buy_routine(self):
        """
        每日买入例程 (下午 14:30 触发)
        """
        # 调用 scan_market 获取候选股
        final_candidates = self.scan_market()
        
        if not final_candidates:
            print("[OneNight] No stocks passed limit up check. Skipping buy.")
            return

        # 4. 执行买入
        # 规则: 最多10只，每只10万
        buy_list = final_candidates[:10]  # 按量比排序的前10个
        
        target_amount = 100000.0
        
        for stock in buy_list:
            symbol = stock['symbol']
            price = stock['price']
            
            # 计算手数 (向下取整到100的倍数)
            if price <= 0: continue
            
            qty = int(target_amount / price / 100) * 100
            if qty == 0:
                print(f"[OneNight] Price too high for {symbol}, cannot buy 1 hand.")
                continue
                
            actual_amount = qty * price
            fees = self.calculate_fees(actual_amount, is_buy=True)
            
            # 记录交易
            self.db.log_trade({
                'symbol': symbol,
                'name': stock['name'],
                'buy_date': datetime.date.today().strftime("%Y-%m-%d"),
                'buy_price': price,
                'quantity': qty,
                'amount': actual_amount,
                'fees': fees
            })
            print(f"[OneNight] BUY {symbol} {stock['name']}: {qty} shares @ {price}")


    def daily_sell_routine(self):
        """
        每日卖出例程 (上午 09:40 触发)
        """
        print(f"[OneNight] Starting daily sell routine at {datetime.datetime.now()}...")
        
        # 1. 获取持仓
        holdings = self.db.get_active_trades()
        if holdings.empty:
            print("[OneNight] No active holdings to sell.")
            return
            
        # 2. 获取最新行情
        symbols = holdings['symbol'].tolist()
        quotes_df = self.fetcher.get_realtime_quotes(symbols)
        
        if quotes_df.empty:
            print("[OneNight] Failed to fetch quotes for selling.")
            return
            
        # 转换以便查询
        quotes_map = {}
        for _, row in quotes_df.iterrows():
            quotes_map[row['代码']] = float(row['最新价'])
            
        # 3. 执行卖出
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        for idx, trade in holdings.iterrows():
            symbol = trade['symbol']
            trade_id = trade['id']
            buy_amount = trade['amount']
            
            current_price = quotes_map.get(symbol)
            if not current_price or current_price == 0:
                print(f"[OneNight] Warning: No price for {symbol}, skipping sell.")
                continue
                
            qty = trade['quantity']
            sell_amount = qty * current_price
            
            # 计算费用 (卖出包含印花税)
            sell_fees = self.calculate_fees(sell_amount, is_buy=False)
            
            # 计算盈亏
            # PnL = 卖出金额 - 买入金额 - 买入费用(已记) - 卖出费用
            # 注意: trade['amount'] 是买入金额
            # trade['fees'] 是买入费用
            
            # 数据库里 trade['fees'] 存的是买入费用
            # 平仓时更新 fees 为 总费用
            
            pnl = sell_amount - buy_amount - trade['fees'] - sell_fees
            pnl_pct = (pnl / buy_amount) * 100 if buy_amount > 0 else 0
            
            self.db.close_trade(trade_id, {
                'sell_date': today_str,
                'sell_price': current_price,
                'sell_amount': sell_amount,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'sell_fees': sell_fees 
            })
            
            print(f"[OneNight] SOLD {symbol}: PnL {pnl:.2f} ({pnl_pct:.2f}%)")

if __name__ == "__main__":
    # Test stub
    db = DatabaseManager()
    fetcher = StockDataFetcher()
    strategy = OneNightStrategy(fetcher, db)
    # strategy.daily_buy_routine()
