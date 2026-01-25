import sys
import os
import datetime
import pandas as pd
import random

# 将当前目录加入路径以便导入
sys.path.append(os.path.dirname(__file__))

from data_fetcher import StockDataFetcher
from one_night_strategy import OneNightStrategy
from database import DatabaseManager

def run_simulation_test():
    print("=== [Full Operation Simulation] Starting 14:30 Trading Day Scan ===")
    
    # 1. 初始化组件
    print("\n[Step 1] Initializing Components...")
    try:
        fetcher = StockDataFetcher()
        db = DatabaseManager()
        strategy = OneNightStrategy(fetcher, db)
        print("✓ Initialization Success.")
    except Exception as e:
        print(f"✗ Initialization Failed: {e}")
        return

    # 2. 检查数据源健康度 (东财 vs 熔断)
    print("\n[Step 2] Checking Primary Source (EastMoney) Health...")
    is_em_ok = fetcher.probe_em_health()
    source_name = "EastMoney" if is_em_ok else "BaoStock/Sina (Circuit Broken)"
    print(f"Current Primary Source for K-line: {source_name}")

    # 3. 执行 14:30 全量扫描与买入逻辑
    # 这个方法内部会完成：拉取实时行情 -> 5项过滤 -> 候选股 K线拉取 -> 涨停验证 -> 写入数据库
    print("\n[Step 3] Executing 14:30 Buy Routine (Full Market Scan & Strategy Application)...")
    try:
        print("Starting strategy.daily_buy_routine(). This may take 1-2 minutes...")
        strategy.daily_buy_routine()
        print("✓ 14:30 Buy Routine Execution Finished.")
    except Exception as e:
        print(f"✗ Error during buy routine: {e}")
        import traceback
        traceback.print_exc()

    # 4. 验证数据库中的买入结果
    print("\n[Step 4] Verifying Simulated Holdings in Database...")
    try:
        holdings = db.get_active_trades()
        if not holdings.empty:
            print(f"✓ Success: Found {len(holdings)} stocks in current holdings:")
            # 对齐显示
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(holdings[['id', 'symbol', 'name', 'buy_date', 'buy_price', 'quantity', 'amount']])
            
            total_investment = holdings['amount'].sum()
            print(f"\nTotal Investment: {total_investment:,.2f} RMB")
        else:
            print("! No stocks matched the strategy criteria today. No trades were logged.")
    except Exception as e:
        print(f"✗ Error reading holdings: {e}")

    # 5. 可选：模拟次日 09:40 卖出逻辑 (带随机波动)
    print("\n[Step 5] Simulating Next-Day 09:40 Sell Routine (with Fake Price Noise)...")
    try:
        holdings = db.get_active_trades()
        if not holdings.empty:
            print(f"Simulating sell for {len(holdings)} stocks with +/- 3% random noise...")
            
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            symbols = holdings['symbol'].tolist()
            quotes_df = fetcher.get_realtime_quotes(symbols)
            
            # 建立价格映射
            quotes_map = {row['代码']: float(row['最新价']) for _, row in quotes_df.iterrows()}
            
            for _, trade in holdings.iterrows():
                symbol = trade['symbol']
                trade_id = trade['id']
                buy_amount = trade['amount']
                buy_fees = trade['fees']
                qty = trade['quantity']
                
                # 获取基准价并加入波动
                base_price = quotes_map.get(symbol, trade['buy_price'])
                noise = 1.0 + random.uniform(-0.03, 0.03) # -3% 到 +3% 波动
                simulated_sell_price = round(base_price * noise, 2)
                
                sell_amount = qty * simulated_sell_price
                # 重新计算卖出手续费 (印花税+佣金)
                sell_fees = strategy.calculate_fees(sell_amount, is_buy=False)
                
                # 计算盈亏
                pnl = sell_amount - buy_amount - buy_fees - sell_fees
                pnl_pct = (pnl / buy_amount) * 100 if buy_amount > 0 else 0
                
                # 更新数据库
                db.close_trade(trade_id, {
                    'sell_date': today_str,
                    'sell_price': simulated_sell_price,
                    'sell_amount': sell_amount,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'sell_fees': sell_fees 
                })
                print(f"  SOLD {symbol}: {trade['buy_price']} -> {simulated_sell_price} (PnL: {pnl:.2f}, {pnl_pct:.2f}%)")

            print("\n[Final Report] Simulated Session Performance Summary:")
            history = db.get_trade_history(limit=len(holdings))
            # 过滤出刚才买入的那些股票（通过 symbol 匹配）
            current_symbols = holdings['symbol'].tolist()
            recent_history = history[history['symbol'].isin(current_symbols)]
            
            if not recent_history.empty:
                print(recent_history[['symbol', 'name', 'buy_price', 'sell_price', 'pnl', 'pnl_pct']])
                total_pnl = recent_history['pnl'].sum()
                print(f"\nTotal Session Net PnL: {total_pnl:,.2f} RMB")
            else:
                print("No recent history found for the traded symbols.")
        else:
            print("Skipped: No holdings to sell.")
    except Exception as e:
        print(f"✗ Error during sell simulation: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== [Full Operation Simulation] All Steps Completed ===")

if __name__ == "__main__":
    run_simulation_test()
