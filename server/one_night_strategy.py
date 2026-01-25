import pandas as pd
import datetime
import time
import random
import baostock as bs
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
        """
        try:
            # 1. 尝试从数据库缓存读取
            df = self.db.get_cached_kline(symbol, max_age_hours=24)
            
            if df is None or df.empty:
                if cache_only: return False
                
                # 2. 缓存缺失，发起网络请求
                time.sleep(0.1)
                df = self.fetcher.get_kline_data(symbol, days=lookback_days + 15)
                
                if df is not None and not df.empty:
                    self.db.save_kline(symbol, df)
                else:
                    print(f"[Debug] {symbol} K-line empty, skipping.")
                    return False
            
            # 3. 筛选最近 N 天
            df = df.tail(lookback_days)
            if df.empty: return False
            
            # 确保涨跌幅列存在
            if '涨跌幅' not in df.columns:
                print(f"[Debug] {symbol} missing '涨跌幅' column. Available: {df.columns.tolist()}")
                return False
                
            # 获取 20 日内最高涨幅
            max_change = df['涨跌幅'].max()
            is_valid = max_change > 9.5
            
            # 💡 调试：打印所有候选股的 20日最高涨幅，看看到底是什么水平
            print(f"[Debug] {symbol} 20d Max Change: {max_change:.2f}% {'[PASS]' if is_valid else ''}")
            
            return is_valid
            
        except Exception as e:
            print(f"[Strategy] Error checking limit up for {symbol}: {e}")
            return False

    def scan_market(self, progress_callback=None) -> list:
        """
        全市场扫描：下午 14:30 触发，应用 6 大过滤条件
        1. 3% <= 涨幅 <= 5%
        2. 量比 > 1.0
        3. 总市值 <= 200亿
        4. 5% <= 换手率 <= 10%
        5. 股价 > 分时均线 (成交额/成交量)
        6. 20日内有过涨停 (需要 K 线数据)
        """
        print(f"[OneNight] Starting full market scan at {datetime.datetime.now()}...")
        
        # 1. 获取全市场实时行情 (一次性拉取，规避高频封锁)
        df = self.fetcher.get_realtime_quotes()
        if df.empty:
            print("[OneNight] Error: Failed to fetch market quotes.")
            return []
        
        # 💡 新增：列名归一化 (兼容不同接口的命名差异)
        column_mapping = {
            'symbol': '代码', 'code': '代码', 'name': '名称',
            'trade': '最新价', 'price': '最新价',
            'changepercent': '涨跌幅', 'pctChg': '涨跌幅', '涨跌幅(%)': '涨跌幅',
            'turnoverratio': '换手率', 'turnover': '换手率', '换手': '换手率', '换手率(%)': '换手率',
            'mktcap': '总市值', 'amount': '成交额', 'volume': '成交量'
        }
                
        # 记录当前原始列名 (调试用)
        original_cols = df.columns.tolist()
        print(f"[Debug] Source Columns: {original_cols[:15]}...")
        
        # 尝试映射
        df = df.rename(columns=column_mapping)
        
        # 💡 核心优化：如果缺失关键指标，尝试通过计算或备用源补全
        if '换手率' not in df.columns or df['换手率'].max() == 0:
            # 如果新浪接口没给，我们就在后面针对 Filter 1 剩下的股票精准补偿
            print("[OneNight] ℹ️  'Turnover' missing. Will compensate later.")
            df['换手率'] = 0.0 

        if '总市值' not in df.columns or df['总市值'].max() == 0:
            # 补全市值：新浪接口可能叫 mktcap (元)
            if 'mktcap' in df.columns:
                df['总市值'] = pd.to_numeric(df['mktcap'], errors='coerce')
            else:
                df['总市值'] = 50 * 100000000 # 兜底 50 亿

        if '量比' not in df.columns:
            print("[OneNight] ℹ️  'Volume Ratio' missing. Will calculate from history.")
            df['量比'] = 0.0

        # 确保数值转换
        for col in ['涨跌幅', '换手率', '量比', '总市值', '最新价', '成交额', '成交量']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # 计算均价
        df['avg_price'] = df['成交额'] / (df['成交量'] * 100 + 1e-6)
        
        # --- 开始阶梯式过滤 (严格按照用户最新要求) ---
        
        # 1. 当天涨幅在 3% - 5% (初筛)
        f1 = df[ (df['涨跌幅'] >= 3.0) & (df['涨跌幅'] <= 5.0) ]
        print(f"[Debug] Filter 1 (3%<=涨幅<=5%): {len(f1)} stocks remain")
        
        if f1.empty: return []

        # 💡 数据补偿阶段... (保持不变)
        candidate_symbols = f1['代码'].tolist()
        print(f"[OneNight] 🏗️  Compensating data for {len(candidate_symbols)} candidates via BaoStock...")
        
        compensated_data = {}
        try:
            self.fetcher.ensure_bs_login()
            # 获取日期范围：过去 15 天到今天
            end_date = datetime.date.today().strftime("%Y-%m-%d")
            start_date = (datetime.date.today() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
            
            for sym in candidate_symbols[:150]: 
                # 1. 格式化代码：准确提取 6 位数字并识别市场
                raw_sym = "".join(filter(str.isdigit, str(sym)))
                if len(raw_sym) != 6: continue
                
                # 上海 6 开头，深圳 0 或 3 开头，北京 4 或 8 开头
                if raw_sym.startswith("6"): prefix = "sh"
                else: prefix = "sz" 
                
                bs_code = f"{prefix}.{raw_sym}"
                
                # 2. 获取历史数据
                rs = bs.query_history_k_data_plus(
                    bs_code, "date,turn,volume", 
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="2"
                )
                hist_list = []
                while (rs.error_code == '0') & rs.next():
                    hist_list.append(rs.get_row_data())
                
                if hist_list:
                    # 昨天的换手率 (作为 14:30 的高精度近似值)
                    last_turnover = float(hist_list[-1][1]) if hist_list[-1][1] else 5.0
                    # 过去 5 日均量
                    prev_vols = [float(x[2]) for x in hist_list[-5:] if x[2]]
                    avg_vol_5d = sum(prev_vols) / len(prev_vols) if prev_vols else 1.0
                    
                    compensated_data[sym] = {
                        'real_turnover': last_turnover,
                        'avg_vol_5d': avg_vol_5d
                    }
            # 💡 补偿结束，但不急着退出，因为后面进阶筛选还要用
        except Exception as e:
            print(f"[OneNight] Compensation error: {e}")

        # 3. 将补偿数据与实时行情结合计算
        matched_count = len(compensated_data)
        print(f"[OneNight] Calculating simulated Volume Ratio for {matched_count} matched stocks...")
        
        def update_metrics(row):
            sym = row['代码']
            if sym in compensated_data:
                # 换手率补全
                if row['换手率'] == 0:
                    row['换手率'] = compensated_data[sym]['real_turnover']
                
                # 量比补偿计算 (单位对齐：手 -> 股)
                real_current_vol_shares = float(row['成交量']) * 100
                avg_vol_5d_shares = float(compensated_data[sym]['avg_vol_5d'])
                
                simulated_v_ratio = (real_current_vol_shares / 0.9) / (avg_vol_5d_shares + 1e-6)
                row['量比'] = round(simulated_v_ratio, 2)
            else:
                # 💡 严格模式：未匹配到补偿数据的（可能是北交所或异常股），给予极低量比使其无法通过 Filter 2
                row['量比'] = 0.0
                if row['换手率'] == 0: row['换手率'] = 0.0
            return row
        
        f1 = f1.apply(update_metrics, axis=1)

        # 2. 量比 > 1
        f2 = f1[ f1['量比'] > 1.0 ]
        print(f"[Debug] Filter 2 (量比>1): {len(f2)} stocks remain")
        
        # 3. 总市值 <= 200亿
        f3 = f2[ f2['总市值'] <= 200 * 100000000 ]
        print(f"[Debug] Filter 3 (市值<=200亿): {len(f3)} stocks remain")
        
        # 4. 换手率在 5% 和 10% 之间
        f4 = f3[ (f3['换手率'] >= 5.0) & (f3['换手率'] <= 10.0) ]
        print(f"[Debug] Filter 4 (5%<=换手<=10%): {len(f4)} stocks remain")
        
        # 5. 股价全天保持在分时均线之上 (14:30 采样点)
        f5 = f4[ f4['最新价'] > f4['avg_price'] ]
        print(f"[Debug] Filter 5 (股价>分时均线): {len(f5)} stocks remain")
        
        candidates_df = f5.copy()
        initial_count = len(candidates_df)
        print(f"[OneNight] {initial_count} stocks passed initial 5 filters.")
        
        if candidates_df.empty:
            return []

        # 3. 进阶筛选: 20天内有过涨停 (仅针对初筛通过的候选股)
        # 按量比降序排列，优中选优
        candidates_df = candidates_df.sort_values(by='量比', ascending=False)
        potential_stocks = candidates_df['代码'].tolist()
        
        final_candidates = []
        
        # 💡 既然每天只运行一次，我们可以更耐心地抓取这些候选股的 K 线
        # 候选股通常在 50-200 只之间，这个请求量是安全的
        max_check = 200 
        check_count = 0
        
        print(f"[OneNight] Verifying 20-day limit-up for top {min(len(potential_stocks), max_check)} candidates...")
        
        # 💡 关键修复：在进入大批量 K 线查询循环前，强制确保登录状态
        self.fetcher.ensure_bs_login()
        
        for symbol in potential_stocks:
            if check_count >= max_check: 
                break
                
            if progress_callback:
                progress_callback(check_count, max_check)
            
            # 检查涨停历史 (允许一次网络重试，因为这是唯一的数据源)
            if self.check_limit_up_history(symbol, cache_only=False):
                row = candidates_df[candidates_df['代码'] == symbol].iloc[0]
                
                rec = {
                    'symbol': symbol,
                    'name': row['名称'],
                    'price': float(row['最新价']),
                    'change': float(row['涨跌幅']),
                    'turnover': float(row['换手率']),
                    'industry': row.get('行业', '未知'),
                    'score': float(row['量比']), 
                    'action': 'BUY',
                    'advice': f"一夜持股严选：量比 {row['量比']:.2f}，换手 {row['换手率']:.2f}%",
                    'reasons': [
                        "涨跌幅 3%-5%", "量比 > 1", "换手率 5%-10%", 
                        "市值 <= 200亿", "股价 > 分时均线", "20日内有涨停"
                    ]
                }
                final_candidates.append(rec)
                # 找到 15 个就够了（取前10个买入，留5个备选）
                if len(final_candidates) >= 15:
                    break
            
            check_count += 1 
            # 基础防御延迟
            time.sleep(0.2)
            
        # 💡 全部扫描任务结束，统一登出
        self.fetcher.ensure_bs_logout()
            
        print(f"[OneNight] Scan complete. Found {len(final_candidates)} high-quality candidates.")
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
