from fastapi import FastAPI, Query
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from data_fetcher import StockDataFetcher
from sentiment_analyzer import SentimentAnalyzer
from strategy_engine import StrategyEngine
from database import DatabaseManager
from industry_matcher import IndustryMatcher
from news_time_decay import NewsTimeDecay
from backtest_evaluator import BacktestEvaluator
import datetime
import asyncio
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd

app = FastAPI(title="StockSearch API")
fetcher = StockDataFetcher()
analyzer = SentimentAnalyzer()
engine = StrategyEngine()
db = DatabaseManager()
industry_matcher = IndustryMatcher()
# 初始化时间衰减处理器：新闻保留3天，考虑交易日
time_decay = NewsTimeDecay(max_age_days=3, trading_days_only=True)
# 初始化回测评估器：止盈+10%，止损-5%，最大持仓30天
backtest_evaluator = BacktestEvaluator(
    take_profit=0.10,
    stop_loss=-0.05,
    max_hold_days=30,
    fixed_periods=[5, 10, 20, 30]
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsScanner:
    """独立的新闻快讯扫描引擎：全天候运行 (每 5 分钟一次)"""
    def __init__(self, db, analyzer):
        self.db = db
        self.analyzer = analyzer
        self.is_running = True
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 86400  # 每24小时清理一次过期新闻

    def scan_loop(self):
        print("[NewsScanner] Background News Scanner Thread Started.")
        while self.is_running:
            try:
                # 1. 拉取全局快讯
                import akshare as ak
                # print("[NewsScanner] Fetching latest global news...")
                news_df = ak.stock_info_global_cls()
                
                # Check 
                if news_df is None or news_df.empty:
                    print("[NewsScanner] No news fetched via akshare.")
                    time.sleep(300)
                    continue

                news_list_raw = news_df.head(40).to_dict(orient="records")
                
                processed_news_input = []
                for n in news_list_raw:
                    processed_news_input.append({
                        "内容": n.get("内容"),
                        "发布时间": n.get("发布时间"),
                        "标题": n.get("标题")
                    })
                
                # 2. 调用 Analyzer 进行并发分析 (支持 LLM)
                # print(f"[NewsScanner] Analyzing {len(processed_news_input)} items...")
                news_pool = self.analyzer.batch_analyze(processed_news_input)
                
                # 3. 持久化到数据库
                # 在保存前，确保时间包含日期
                current_date = datetime.datetime.now().date()
                news_to_save = []
                for item in news_pool:
                    publish_time = item.get("发布时间", "")
                    # 如果时间只有时间没有日期，添加当前日期
                    if publish_time:
                        publish_time_str = str(publish_time)
                        if ':' in publish_time_str and len(publish_time_str.split(':')) >= 2:
                            if len(publish_time_str) <= 8 and '-' not in publish_time_str:
                                # 只有时间，添加当前日期
                                publish_time = f"{current_date} {publish_time_str}"
                    
                    news_to_save.append({
                        "title": item.get("标题"),
                        "content": item.get("内容"),
                        "time": publish_time,
                        "sentiment": item.get("sentiment")
                    })
                self.db.save_news_batch(news_to_save)
                print(f"[NewsScanner] {len(news_to_save)} news items synced to DB.")
                
                # 定期清理过期新闻（每24小时一次）
                current_time = time.time()
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    try:
                        deleted_count = self.db.cleanup_old_news(max_age_days=7)
                        if deleted_count > 0:
                            print(f"[NewsScanner] Cleaned up {deleted_count} old news items.")
                        self.last_cleanup_time = current_time
                    except Exception as e:
                        print(f"[NewsScanner] Error during cleanup: {e}")
                
                # 每 5 分钟轮询一次
                time.sleep(300)
            except Exception as e:
                print(f"[NewsScanner] Error in loop: {e}")
                time.sleep(60)

class BackgroundScanner:
    """后台异步扫描引擎：负责全市场自动寻迹"""
    def __init__(self, fetcher, engine, db):
        self.fetcher = fetcher
        self.engine = engine
        self.db = db
        self.latest_results = []
        self.is_running = True
        self.scan_count = 0
        self.last_scan_date = "" 
        self.reset_event = threading.Event()

        # 启动自检：尝试从数据库恢复今日已有的扫描快照
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        historical_results = self.db.get_daily_scan(today_str)
        if historical_results:
            print(f"[Scanner] Startup: Restored {len(historical_results)} results for {today_str} from database.")
            self.latest_results = historical_results
            self.last_scan_date = today_str

    def trigger_scan(self):
        """手动外部唤醒扫描 (如权重变更后)，重置日期强制重扫"""
        print("[Scanner] Signal received: Resetting date and triggering immediate FULL-MARKET re-scan...")
        self.last_scan_date = "" 
        self.reset_event.set()

    def scan_loop(self):
        print("[Scanner] Background Full-Market Scanner Thread Started.")
        while self.is_running:
            try:
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                
                now = datetime.datetime.now()
                
                # 如果今天已经扫过了，且没有收到强制重扫信号，就进入长休眠
                if self.last_scan_date == today_str:
                    print(f"[Scanner] Today's scan ({today_str}) already complete. Waiting for reset or next day...")
                    if self.reset_event.wait(3600): # 每小时检查一次，除非被 reset_event 唤醒
                         print("[Scanner] Manual trigger detected. Restarting scan...")
                    self.reset_event.clear()
                    continue

                # 修正：跨天后必须等到收盘后 (15:00) 才能获取当日全量数据
                # 否则获取的是昨日数据，且会导致今日后续真正收盘时不再运行
                if now.hour < 15 and not self.reset_event.is_set():
                    print(f"[Scanner] It's {now.strftime('%H:%M')}, market not closed yet. Waiting for 15:00 to start Daily Scan...")
                    if self.reset_event.wait(1800): # 每 30 分钟检查一次，支持手动触发例外
                        print("[Scanner] Manual trigger detected. Force starting scan...")
                    else:
                        continue

                self.scan_count += 1
                print(f"\n[Scanner] === Starting FULL-MARKET Scan #{self.scan_count} ({today_str}) ===")
                
                # 1. 获取全市场标的快照 (5000+)
                full_spot = self.fetcher.get_realtime_quotes([])
                if full_spot.empty:
                    print("[Scanner] Warning: Could not fetch market spot. Retrying in 60s...")
                    time.sleep(60)
                    continue
                
                # 过滤垃圾股 (成交额太小的僵尸股不进 K 线程，节省流量)
                valid_spot = full_spot[full_spot['成交额'] > 5000000] # 500w 以上成交额
                all_symbols = valid_spot['代码'].tolist()
                print(f"[Scanner] Market snapshot loaded. Filtering to {len(all_symbols)} active candidates.")

                new_recommendations = []
                weights = self.db.get_weights()
                
                # 从数据库获取最新新闻池用于辅助打分 (Read-Only)
                # 只获取最近3天的新闻，并应用时间衰减
                news_pool = []
                try:
                    db_news = self.db.get_latest_news(limit=100, max_age_days=3)  # 获取更多新闻，然后过滤
                    for dn in db_news:
                         news_pool.append({
                             "content": dn['content'],
                             "title": dn['title'], 
                             "time": dn.get('time', ''),
                             "sentiment": dn['sentiment'],
                             "created_at": dn.get('created_at', '')
                         })
                    
                    # 应用时间衰减过滤（过滤过期新闻并计算权重）
                    news_pool = time_decay.filter_news_by_age(news_pool)
                    
                    print(f"[Scanner] Loaded {len(news_pool)} valid news items (within 3 trading days) from DB for context awareness.")
                except Exception as e:
                    print(f"[Scanner] Error loading news from DB: {e}")
                
                
                # 定义单股处理函数供线程池调用
                def process_stock(symbol):
                    try:
                        # 0. 优先从数据库缓存中读取 K 线数据 (有效期 1 天)
                        kline = self.db.get_cached_kline(symbol)
                        if kline is None:
                            kline = self.fetcher.get_kline_data(symbol, days=150)
                            if not kline.empty:
                                self.db.save_kline(symbol, kline)
                        
                        if kline.empty: return None
                        
                        # 1. 优先从本地数据库读取个股画像 (避免重复请求 ak.stock_individual_info_em)
                        stock_info = self.db.get_cached_stock_info(symbol)
                        if not stock_info:
                            stock_info = self.fetcher.get_stock_info(symbol)
                            if stock_info:
                                self.db.save_stock_info(symbol, stock_info)
                        
                        stock_name = stock_info.get('股票简称', symbol)
                        stock_industry = stock_info.get('行业', '未知')
                        
                        # 2. 获取周线数据（用于多时间框架确认）
                        weekly_kline = None
                        try:
                            weekly_kline = self.fetcher.get_kline_data(symbol, period="weekly", days=200)
                            if weekly_kline.empty:
                                weekly_kline = None
                        except:
                            weekly_kline = None
                        
                        # 3. 获取财务数据（用于基本面因子）
                        finance_data = None
                        try:
                            finance_list = self.fetcher.get_company_finance(symbol)
                            if finance_list and len(finance_list) > 0:
                                finance_data = finance_list[0]
                        except:
                            finance_data = None
                        
                        # 4. 情感评估 - 使用行业匹配器进行智能匹配
                        # 不仅检查股票名称是否在新闻中出现，还检查新闻分析的行业是否与股票行业匹配
                        dynamic_sent = 0.0
                        
                        if news_pool:
                            # 使用行业匹配器获取相关新闻
                            # news_pool 结构: {content, title, sentiment}
                            matching_news = industry_matcher.get_matching_news(
                                news_pool,
                                stock_industry=stock_industry,
                                stock_name=stock_name,
                                symbol=symbol
                            )
                            
                            if matching_news:
                                # 使用行业匹配器计算情感得分（集成时间衰减）
                                # 直接提及的新闻权重更高，行业匹配的新闻权重较低
                                # 时间衰减会自动应用到每条新闻的得分上
                                dynamic_sent = industry_matcher.calculate_sentiment_score(
                                    matching_news,
                                    direct_mention_weight=0.3,  # 直接提及每条0.3分
                                    industry_match_weight=0.15,  # 行业匹配每条0.15分
                                    time_decay_handler=time_decay  # 应用时间衰减
                                )
                                # 限制在合理范围内
                                dynamic_sent = max(-1.0, min(1.0, dynamic_sent))

                        # 5. 生成推荐（使用多因子模型）
                        rec = self.engine.generate_recommendation(
                            kline, 
                            dynamic_sent,
                            tech_weight=weights.get('tech_weight', 0.6),  # 技术因子权重
                            sentiment_weight=weights.get('sentiment_weight', 0.2),  # 消息面权重
                            fundamental_weight=weights.get('fundamental_weight', 0.15),  # 基本面权重
                            risk_weight=weights.get('risk_weight', 0.05),  # 风险因子权重
                            finance_data=finance_data,  # 财务数据
                            weekly_kline_df=weekly_kline  # 周线数据
                        )
                        
                        rec.update({
                            'symbol': symbol,
                            'name': stock_name,
                            'price': float(rec.get('price', 0)),
                            'industry': stock_industry,
                            'change': float(stock_info.get('涨跌幅', 0)),
                            'turnover': float(stock_info.get('换手率', 0))
                        })
                        
                        # 只保存BUY和HOLD的推荐到数据库（用于回测）
                        if rec['action'] in ["BUY", "HOLD"]:
                            # 添加行业和因子得分信息
                            rec_for_db = rec.copy()
                            rec_for_db['industry'] = stock_industry
                            rec_for_db['factor_scores'] = rec.get('factor_scores', {})
                            self.db.save_recommendation(rec_for_db)
                        
                        return rec
                    except Exception as e:
                        # print(f"[Scanner] Error processing {symbol}: {e}") # Too verbose
                        return None

                # 2. 并发深度扫描 (限制 10 线程保护 API 负载)
                print(f"[Scanner] Starting Concurrent Evaluation with 10 threads...")
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(process_stock, sym): sym for sym in all_symbols}
                    done_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            new_recommendations.append(res)
                            # 只保存BUY和HOLD的推荐到数据库（用于回测）
                            if res['action'] in ["BUY", "HOLD"]:
                                # 确保包含行业和因子得分信息
                                rec_for_db = res.copy()
                                if 'industry' not in rec_for_db:
                                    rec_for_db['industry'] = '未知'
                                if 'factor_scores' not in rec_for_db:
                                    rec_for_db['factor_scores'] = {}
                                self.db.save_recommendation(rec_for_db)
                        
                        done_count += 1
                        if done_count % 100 == 0:
                            print(f"[Scanner] Progress: {done_count}/{len(all_symbols)} evaluated...")

                # 3. 更新内存缓存 (按评分排序，取前 12 个)
                if not new_recommendations:
                    print("[Scanner] Warning: All evaluations failed. Results empty.")
                    self.latest_results = []
                else:
                    # 将所有评估结果按评分排序
                    full_ranks = sorted(new_recommendations, key=lambda x: x['score'], reverse=True)
                    self.latest_results = full_ranks[:12] # 主页依然显示 Top 12
                    self.last_scan_date = today_str 
                    # 将全部排序结果存入数据库供排行榜分页查阅
                    self.db.save_daily_scan(today_str, full_ranks)
                    print(f"[Scanner] Full Scan Complete. Saved {len(full_ranks)} stocks to Database for Market Ranking.")
                
                self.reset_event.clear()
            except Exception as e:
                print(f"[Scanner] Critical Error in Loop: {e}")
                time.sleep(60)

# 初始化并启动后台扫描引擎
# 初始化并启动后台扫描引擎
scanner = BackgroundScanner(fetcher, engine, db)
threading.Thread(target=scanner.scan_loop, daemon=True).start()

# 启动独立的新闻扫描引擎
news_scanner = NewsScanner(db, analyzer)
threading.Thread(target=news_scanner.scan_loop, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "StockSearch API is running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/stocks/realtime")
def get_realtime_stocks(symbols: Optional[str] = Query(None)):
    """获取实时行情，symbols 以逗号分隔"""
    symbol_list = symbols.split(",") if symbols else []
    df = fetcher.get_realtime_quotes(symbol_list)
    return df.to_dict(orient="records")

@app.get("/api/stocks/kline/{symbol}")
def get_stock_kline(symbol: str, period: str = "daily", days: int = 200):
    """获取 K 线历史数据"""
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y%m%d")
    df = fetcher.get_kline_data(symbol, period=period, start_date=start_date)
    return df.to_dict(orient="records")

@app.get("/api/stocks/info/{symbol}")
def get_stock_info(symbol: str):
    """获取个股基本面基础信息 (名称、行业等)"""
    return fetcher.get_stock_info(symbol)

@app.get("/api/stocks/finance/{symbol}")
def get_stock_finance(symbol: str):
    """获取公司核心财务指标"""
    return fetcher.get_company_finance(symbol)

@app.get("/api/news/flash")
def get_news_flash():
    """获取最新财经快讯 (优先读库，实现毫秒级响应)"""
    try:
        # 1. 尝试从数据库读取现成结果
        cached_news = db.get_latest_news(limit=20)
        if cached_news:
            return cached_news
            
        # 2. 如果库为空 (比如刚启动还没运行 scanner)，则降级为实时拉取
        # 注意：这会比较慢，因为包含 LLM 调用
        import akshare as ak
        import pandas as pd
        news_df = ak.stock_info_global_cls()
        news_df = news_df.rename(columns={
            "发布时间": "time",
            "内容": "content",
            "标题": "title"
        })
        # 确保按时间降序排序（最新的在前）
        if "time" in news_df.columns and not news_df.empty:
            # 尝试将时间转换为 datetime 进行排序
            try:
                current_date = datetime.datetime.now().date()
                # 如果时间只有时间没有日期，添加当前日期
                def add_date_if_needed(time_str):
                    if pd.isna(time_str) or not time_str:
                        return None
                    time_str = str(time_str)
                    # 如果只有时间（长度 <= 8，如 "14:30:00"），添加当前日期
                    if ':' in time_str and len(time_str) <= 8 and '-' not in time_str:
                        return f"{current_date} {time_str}"
                    return time_str
                
                news_df['time'] = news_df['time'].apply(add_date_if_needed)
                news_df['time'] = pd.to_datetime(news_df['time'], errors='coerce')
                news_df = news_df.sort_values('time', ascending=False)
                # 转换回字符串格式以便返回（包含完整日期时间）
                news_df['time'] = news_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(f"[News] Error processing time: {e}")
                # 如果转换失败，按字符串降序排序
                news_df = news_df.sort_values('time', ascending=False)
        
        news_list = news_df.head(20).to_dict(orient="records")
        processed_news = analyzer.batch_analyze(news_list)
        
        # 顺便存入库
        db.save_news_batch(processed_news)
        
        # 返回的列表已经按时间降序排序（最新的在前）
        return processed_news
    except Exception as e:
        print(f"News error: {e}")
        return {"error": str(e)}

@app.get("/api/stocks/recommend/{symbol}")
def get_stock_recommendation(symbol: str):
    """【单股诊断】获取指定个股的核心策略建议并持久化记录"""
    try:
        kline_df = fetcher.get_kline_data(symbol, days=250)
        if kline_df.empty:
            return {"error": f"无法获取股票 {symbol} 的数据"}
            
        sentiment_score = 0.1 
        # 获取当前动态权重
        weights = db.get_weights()
        
        # 获取财务数据
        finance_data = None
        try:
            finance_list = fetcher.get_company_finance(symbol)
            if finance_list and len(finance_list) > 0:
                finance_data = finance_list[0]
        except:
            finance_data = None
        
        # 获取周线数据
        weekly_kline = None
        try:
            weekly_kline = fetcher.get_kline_data(symbol, period="weekly", days=200)
            if weekly_kline.empty:
                weekly_kline = None
        except:
            weekly_kline = None
        
        recommendation = engine.generate_recommendation(
            kline_df, 
            sentiment_score,
            tech_weight=weights.get('tech_weight', 0.6),
            sentiment_weight=weights.get('sentiment_weight', 0.2),
            fundamental_weight=weights.get('fundamental_weight', 0.15),
            risk_weight=weights.get('risk_weight', 0.05),
            finance_data=finance_data,
            weekly_kline_df=weekly_kline
        )
        
        # 补全代码字段
        recommendation['symbol'] = symbol
        # 强制转换价格为 float，防止数据库报错
        recommendation['price'] = float(recommendation.get('price', 0))
        
        # 获取行业信息
        stock_info = fetcher.get_stock_info(symbol)
        recommendation['industry'] = stock_info.get('行业', '未知')
        
        # 保存到数据库（只保存BUY和HOLD）
        if recommendation['action'] in ["BUY", "HOLD"]:
            db.save_recommendation(recommendation)
        
        return recommendation
    except Exception as e:
        print(f"Error diagnosing {symbol}: {e}")
        return {"error": str(e)}

@app.get("/api/stocks/market_recommendations")
def get_market_recommendations():
    """【主页看板】返回 Top 12 核心推荐"""
    return scanner.latest_results[:12]

@app.get("/api/stocks/full_rank")
def get_full_market_rank(
    page: int = Query(1, ge=1), 
    page_size: int = Query(500, ge=1, le=1000),
    search: str = Query(None),
    industry: str = Query(None),
    min_price: float = Query(None),
    max_price: float = Query(None),
    sort_by: str = Query('score'),
    sort_dir: str = Query('desc')
):
    """【排行榜页】返回当日全量扫描结果，支持分页、搜索、筛选和排序"""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    # 优先从数据库加载今日完整名单
    ranks = db.get_daily_scan(today_str)
    if not ranks:
        # 如果数据库还没写完，尝试由内存返回
        ranks = scanner.latest_results
    
    if not ranks:
        return {"data": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}
    
    # 1. 应用过滤
    filtered_ranks = ranks
    
    # 搜索 (Symbol / Name)
    if search:
        search_lower = search.lower()
        filtered_ranks = [
            r for r in filtered_ranks 
            if search_lower in str(r.get('symbol', '')).lower() or search_lower in str(r.get('name', '')).lower()
        ]
        
    # 行业筛选
    if industry:
        filtered_ranks = [r for r in filtered_ranks if r.get('industry') == industry]
        
    # 价格区间
    if min_price is not None:
        filtered_ranks = [r for r in filtered_ranks if float(r.get('price', 0)) >= min_price]
        
    if max_price is not None:
        filtered_ranks = [r for r in filtered_ranks if float(r.get('price', 0)) <= max_price]
        
    # 2. 应用排序
    reverse = (sort_dir == 'desc')
    try:
        # 处理可能缺失的字段，设置默认值
        def get_sort_key(item):
            val = item.get(sort_by)
            if val is None:
                return -float('inf') if reverse else float('inf')
            try:
                return float(val) # 尝试转为数字
            except (ValueError, TypeError):
                return str(val) # 否则按字符串排序

        filtered_ranks.sort(key=get_sort_key, reverse=reverse)
    except Exception as e:
        print(f"Sort error: {e}")
        # Fallback to score sort if custom sort fails
        filtered_ranks.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # 3. 计算分页
    total = len(filtered_ranks)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    
    # 4. 获取当前页数据
    page_data = filtered_ranks[start_idx:end_idx]
    
    return {
        "data": page_data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@app.get("/api/stocks/industry_distribution")
def get_industry_distribution(limit: int = Query(200, ge=10, le=2000)):
    """统计全场排名前 N 名股票的行业分布（饼图数据）"""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    ranks = db.get_daily_scan(today_str)
    if not ranks:
        ranks = scanner.latest_results
        
    if not ranks:
        return []
        
    # 按分数排序确保是 Top N
    sorted_stocks = sorted(ranks, key=lambda x: x.get('score', 0), reverse=True)
    top_stocks = sorted_stocks[:limit]
    
    # 统计
    industries = [s.get('industry', '其他') for s in top_stocks if s.get('industry')]
    counter = Counter(industries)
    
    # 取前 9 个行业，其他的归为 "其他"
    top_industries = counter.most_common(9)
    other_count = len(top_stocks) - sum(item[1] for item in top_industries)
    
    result = [{"name": k, "value": v} for k, v in top_industries]
    if other_count > 0:
        result.append({"name": "其他", "value": other_count})
        
    return result

@app.get("/api/stocks/industries")
def get_all_industries():
    """获取全市场所有存在的行业列表（用于筛选）"""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    ranks = db.get_daily_scan(today_str)
    if not ranks:
        ranks = scanner.latest_results
        
    if not ranks:
        return []
        
    # 提取所有不为空的行业并去重
    industries = sorted(list(set(r.get('industry') for r in ranks if r.get('industry'))))
    return industries

@app.post("/api/admin/trigger_scan")
def trigger_manual_scan():
    """【管理端点】手动触发全量市场扫描"""
    try:
        # 清除今日扫描标记，强制重新扫描
        scanner.last_scan_date = None
        scanner.reset_event.set()
        return {"status": "success", "message": "Full market scan triggered. Please wait 5-10 minutes for completion."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/review/performance")
def get_performance_review():
    """【绩效复盘】获取所有已关闭或活跃的推荐订单表现（使用新的回测逻辑）"""
    try:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        # 获取所有OPEN的推荐
        open_recs = db.get_open_recommendations()
        if not open_recs.empty:
            try:
                symbols = open_recs['symbol'].unique().tolist()
                quotes = fetcher.get_realtime_quotes(symbols)
            except Exception as e:
                print(f"[Performance] Error fetching quotes: {e}")
                quotes = pd.DataFrame()
            
            for idx, rec in open_recs.iterrows():
                try:
                    symbol = rec.get('symbol') if 'symbol' in rec.index else None
                    if not symbol:
                        continue
                    
                    matching_quote = quotes[quotes['代码'] == symbol] if not quotes.empty else pd.DataFrame()
                    if not matching_quote.empty:
                        current_price = float(matching_quote.iloc[0]['最新价'])
                        # 使用 pandas Series 的索引方式（Series 支持 .get() 方法）
                        try:
                            entry_price = rec.get('entry_price', None)
                            if entry_price is None or (isinstance(entry_price, (int, float)) and (pd.isna(entry_price) or entry_price == 0)):
                                entry_price = rec.get('price', None)
                            if entry_price is None or (isinstance(entry_price, (int, float)) and (pd.isna(entry_price) or entry_price == 0)):
                                entry_price = current_price
                            entry_price = float(entry_price)
                        except (ValueError, TypeError, KeyError):
                            entry_price = current_price
                        
                        try:
                            entry_date = rec.get('entry_date', None)
                            if entry_date is None or (isinstance(entry_date, str) and len(entry_date) == 0):
                                entry_date = rec.get('created_at', None)
                            if entry_date is None or (isinstance(entry_date, str) and len(entry_date) == 0):
                                entry_date = today_str
                            # 确保是字符串
                            if not isinstance(entry_date, str):
                                entry_date = str(entry_date)
                        except (ValueError, TypeError, KeyError):
                            entry_date = today_str
                        
                        # 获取当前action（重新评估）
                        current_action = None
                        try:
                            kline_df = fetcher.get_kline_data(symbol, days=150)
                            if not kline_df.empty:
                                # 重新生成推荐，获取当前action
                                temp_rec = engine.generate_recommendation(kline_df, sentiment_score=0.0)
                                current_action = temp_rec.get('action')
                        except Exception as e:
                            print(f"[Performance] Error re-evaluating {symbol}: {e}")
                            pass
                        
                        # 判断是否应该卖出
                        try:
                            # 确保 entry_price 是 float
                            entry_price_float = float(entry_price) if entry_price else current_price
                            # 确保 entry_date 是字符串格式
                            if isinstance(entry_date, str):
                                entry_date_str = entry_date[:10] if len(entry_date) >= 10 else entry_date
                            else:
                                entry_date_str = str(entry_date)[:10] if entry_date else today_str
                            
                            exit_decision = backtest_evaluator.should_exit(
                                entry_price=entry_price_float,
                                current_price=current_price,
                                entry_date=entry_date_str,
                                current_date=today_str,
                                current_action=current_action
                            )
                        except Exception as e:
                            print(f"[Performance] Error in should_exit for {symbol}: {e}")
                            continue
                        
                        if exit_decision['should_exit']:
                            try:
                                # 计算最大盈利/亏损（简化版，实际应该从历史价格计算）
                                max_profit = max(0, exit_decision['pnl']) if exit_decision['pnl'] > 0 else None
                                max_loss = min(0, exit_decision['pnl']) if exit_decision['pnl'] < 0 else None
                                
                                # 计算相对市场收益（简化版，实际需要市场指数数据）
                                relative_return = None  # 暂时不计算，需要市场指数数据
                                
                                # 获取记录ID
                                rec_id = rec.get('id') if 'id' in rec.index else None
                                if rec_id:
                                    # 更新推荐记录
                                    db.update_recommendation(
                                        rec_id,
                                        exit_decision['exit_price'],
                                        exit_decision['pnl'],
                                        exit_reason=exit_decision['exit_reason'],
                                        max_profit=max_profit,
                                        max_loss=max_loss,
                                        relative_return=relative_return
                                    )
                            except Exception as e:
                                print(f"[Performance] Error updating recommendation for {symbol}: {e}")
                                continue
                except Exception as e:
                    print(f"[Performance] Error processing record {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 获取所有已关闭的推荐
        try:
            performance = db.get_performance_stats()
            
            # 计算绩效指标
            if not performance.empty and len(performance) > 0:
                # 确保必要的列存在
                if 'pnl' not in performance.columns:
                    performance['pnl'] = 0.0
                if 'score' not in performance.columns:
                    performance['score'] = 0.0
                if 'hold_days' not in performance.columns:
                    performance['hold_days'] = None
                
                # 处理 NaN 值
                performance['pnl'] = performance['pnl'].fillna(0.0)
                performance['score'] = performance['score'].fillna(0.0)
                
                try:
                    metrics = backtest_evaluator.calculate_performance_metrics(performance)
                except Exception as e:
                    print(f"[Performance] Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    metrics = {}
                
                # 只有当有 score 列时才计算分组指标
                grouped_metrics = {}
                if 'score' in performance.columns and not performance['score'].isna().all():
                    try:
                        grouped_metrics = backtest_evaluator.calculate_grouped_metrics(performance, group_by='score_range')
                    except Exception as e:
                        print(f"[Performance] Error calculating grouped metrics: {e}")
                        import traceback
                        traceback.print_exc()
                        grouped_metrics = {}
                
                try:
                    # 清理NaN值，确保JSON可序列化
                    performance_clean = performance.copy()
                    # 将所有NaN替换为None（JSON会转换为null）
                    performance_clean = performance_clean.where(pd.notna(performance_clean), None)
                    
                    # 转换为字典并清理NaN
                    performance_dict = performance_clean.to_dict(orient="records")
                    # 递归清理字典中的NaN值
                    def clean_nan(obj):
                        if isinstance(obj, dict):
                            return {k: clean_nan(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [clean_nan(item) for item in obj]
                        elif isinstance(obj, float) and (pd.isna(obj) or np.isnan(obj) or np.isinf(obj)):
                            return None
                        return obj
                    
                    performance_dict = clean_nan(performance_dict)
                    metrics_clean = clean_nan(metrics)
                    grouped_metrics_clean = clean_nan(grouped_metrics)
                    
                    return {
                        'performance_data': performance_dict,
                        'overall_metrics': metrics_clean,
                        'grouped_metrics': grouped_metrics_clean
                    }
                except Exception as e:
                    print(f"[Performance] Error converting to dict: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        'performance_data': [],
                        'overall_metrics': {},
                        'grouped_metrics': {}
                    }
            else:
                return {
                    'performance_data': [],
                    'overall_metrics': {},
                    'grouped_metrics': {}
                }
        except Exception as e:
            print(f"[Performance] Error in get_performance_review: {e}")
            import traceback
            traceback.print_exc()
            return {
                'performance_data': [],
                'overall_metrics': {},
                'grouped_metrics': {},
                'error': str(e)
            }
    except Exception as e:
        print(f"[Performance] Fatal error in get_performance_review: {e}")
        import traceback
        traceback.print_exc()
        return {
            'performance_data': [],
            'overall_metrics': {},
            'grouped_metrics': {},
            'error': str(e)
        }

@app.get("/api/review/iterate")
def iterate_strategy():
    """【自迭代】根据历史胜率自动优化模型权重"""
    # 1. 获取所有历史复盘数据
    performance = db.get_performance_stats()
    if performance.empty or len(performance) < 5:
        return {"status": "skipped", "reason": "样本不足，至少需要5条复盘数据才能启动迭代"}
    
    # 2. 计算胜率
    win_rate = len(performance[performance['pnl'] > 0]) / len(performance)
    avg_pnl = performance['pnl'].mean()
    
    # 3. 简单的迭代逻辑 (启发式)
    # 如果近期的胜率低于 50% 但消息面得分较高时收益好，则增加消息面权重
    weights = db.get_weights()
    current_tech = weights.get('tech_weight', 0.8)
    current_sent = weights.get('sentiment_weight', 0.2)
    
    if avg_pnl < 0:
        # 如果最近亏损，尝试微调权重（探索性更新）
        new_sent = min(0.4, current_sent + 0.05)
        new_tech = 1.0 - new_sent
        db.update_weight('sentiment_weight', new_sent)
        db.update_weight('tech_weight', new_tech)
        return {
            "status": "optimized",
            "win_rate": f"{win_rate*100:.2f}%",
            "new_weights": {"tech": new_tech, "sentiment": new_sent},
            "msg": "由于近期胜率波动，已自动增加消息面反馈权重以对冲趋势滞后性。"
        }
    
    return {
        "status": "stable",
        "win_rate": f"{win_rate*100:.2f}%", 
        "weights": weights,
        "msg": "当前策略表现稳健，参数无需调整。"
    }

@app.get("/api/review/weights")
def get_current_weights():
    """获取当前策略权重参数"""
    return db.get_weights()

@app.post("/api/review/update_weights")
def update_manual_weights(data: dict):
    """手动保存策略权重参数"""
    try:
        tech = float(data.get('tech_weight', 0.8))
        sent = float(data.get('sentiment_weight', 0.2))
        
        # 确保总和为 1 且在合理区间
        old_weights = db.get_weights()
        # 使用更稳健的精度比对，防止前端传参的小数点误差触发重扫
        if abs(old_weights.get('tech_weight', 0) - tech) < 0.001 and \
           abs(old_weights.get('sentiment_weight', 0) - sent) < 0.001:
            return {"status": "success", "message": "权重参数未实质变更，已跳过全量重扫"}

        db.update_weight('tech_weight', tech)
        db.update_weight('sentiment_weight', sent)
        
        # 联动：立即开启一轮新权重的扫描
        scanner.trigger_scan()
        
        return {"status": "success", "new_weights": {"tech_weight": tech, "sentiment_weight": sent}, "message": "权重已更新，全量重扫已启动"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
