import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from io import StringIO
import os
import gzip

# 支持从环境变量读取数据库路径（Railway部署时使用）
# Railway Volume 挂载路径通常是 /data 或通过环境变量指定
volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/data")
# 如果 /data 目录不存在，则使用当前目录（本地开发）
if not os.path.exists(volume_path):
    volume_path = os.path.dirname(__file__)

# 优先使用环境变量指定的路径，否则使用 Volume 路径或当前目录
DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(volume_path, "stock_logic.db"))

class DatabaseManager:
    """管理回测记录与策略参数的持久化"""
    
    def __init__(self):
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(DB_PATH, timeout=60) # 增加到 60s 超时
        # 启用 WAL 模式提升并发性能
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except:
            pass
        return conn

    def init_db(self):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 1. 推荐历史表（扩展字段用于回测）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    action TEXT,
                    price REAL,
                    score REAL,
                    reasons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'OPEN',
                    close_price REAL,
                    pnl REAL,
                    -- 新增字段用于回测
                    entry_date TEXT,           -- 买入日期（推荐日期）
                    entry_price REAL,          -- 买入价格（推荐日收盘价）
                    exit_date TEXT,            -- 卖出日期
                    exit_price REAL,           -- 卖出价格
                    hold_days INTEGER,         -- 持仓天数
                    exit_reason TEXT,          -- 卖出原因（止盈/止损/时间/信号反转）
                    max_profit REAL,           -- 持仓期间最大盈利（百分比）
                    max_loss REAL,             -- 持仓期间最大亏损（百分比）
                    relative_return REAL,      -- 相对市场收益（vs 沪深300）
                    industry TEXT,             -- 行业（用于分组统计）
                    factor_scores_json TEXT    -- 因子得分（JSON格式，用于分析）
                )
            ''')
            
            # 添加新字段到已存在的表（如果表已存在）
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN entry_date TEXT")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN entry_price REAL")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN exit_date TEXT")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN exit_price REAL")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN hold_days INTEGER")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN exit_reason TEXT")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN max_profit REAL")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN max_loss REAL")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN relative_return REAL")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN industry TEXT")
            except:
                pass
            try:
                cursor.execute("ALTER TABLE recommendations ADD COLUMN factor_scores_json TEXT")
            except:
                pass
            # 2. 策略参数权重表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_params (
                    key TEXT PRIMARY KEY,
                    value REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute("INSERT OR IGNORE INTO strategy_params (key, value) VALUES ('tech_weight', 0.6)")
            cursor.execute("INSERT OR IGNORE INTO strategy_params (key, value) VALUES ('sentiment_weight', 0.2)")
            cursor.execute("INSERT OR IGNORE INTO strategy_params (key, value) VALUES ('fundamental_weight', 0.15)")
            cursor.execute("INSERT OR IGNORE INTO strategy_params (key, value) VALUES ('risk_weight', 0.05)")

            # 3. 每日全量快照表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_scans (
                    scan_date TEXT PRIMARY KEY,
                    results_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 4. 个股画像缓存表 (名称/行业 永久缓存)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_info_cache (
                    symbol TEXT PRIMARY KEY,
                    data_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 5. K 线历史数据缓存 (减少重复抓取)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kline_cache (
                    symbol TEXT PRIMARY KEY,
                    data_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 6. 新闻快讯分析缓存表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_analysis (
                    title_hash TEXT PRIMARY KEY, -- 使用标题+时间的哈希作为主键去重
                    title TEXT,
                    content TEXT,
                    publish_time TEXT,
                    sentiment_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 7. 行业情绪值周累积表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS industry_sentiment_weekly (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    industry TEXT NOT NULL,
                    week_start_date TEXT NOT NULL,  -- 周开始日期 (YYYY-MM-DD)
                    sentiment_score REAL DEFAULT 1.0,  -- 情绪值，默认1.0
                    news_count INTEGER DEFAULT 0,  -- 该周该行业的新闻数量
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(industry, week_start_date)
                )
            ''')
            
            # 8. 一夜持股策略交易记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS one_night_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    name TEXT,
                    buy_date TEXT,      -- 买入日期 YYYY-MM-DD
                    buy_price REAL,     -- 买入价格
                    quantity INTEGER,   -- 买入数量
                    amount REAL,        -- 买入金额 (quantity * buy_price)
                    sell_date TEXT,     -- 卖出日期
                    sell_price REAL,    -- 卖出价格
                    sell_amount REAL,   -- 卖出金额
                    pnl REAL,           -- 盈亏额 (sell_amount - amount - fees)
                    pnl_pct REAL,       -- 盈亏率
                    fees REAL,          -- 手续费总额
                    status TEXT DEFAULT 'HOLD', -- HOLD / SOLD
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_status 
                ON one_night_trades(status)
            ''')
            
            # 创建索引提高查询效率
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_industry_week 
                ON industry_sentiment_weekly(industry, week_start_date)
            ''')
            
            conn.commit()
        finally:
            conn.close()

    def save_recommendation(self, rec: dict):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO recommendations (symbol, action, price, score, reasons)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                rec['symbol'], rec['action'], rec['price'], 
                rec['score'], ",".join(rec['reasons'])
            ))
            conn.commit()
        finally:
            conn.close()

    def get_open_recommendations(self):
        conn = self.get_connection()
        try:
            return pd.read_sql_query("SELECT * FROM recommendations WHERE status = 'OPEN'", conn)
        finally:
            conn.close()

    def update_recommendation(
        self, 
        rec_id, 
        exit_price, 
        pnl, 
        exit_reason=None,
        max_profit=None,
        max_loss=None,
        relative_return=None
    ):
        """更新推荐记录（卖出时）"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            exit_date = datetime.now().strftime("%Y-%m-%d")
            
            # 计算持仓天数
            cursor.execute("SELECT entry_date FROM recommendations WHERE id = ?", (rec_id,))
            row = cursor.fetchone()
            hold_days = None
            if row and row[0]:
                try:
                    entry_date = datetime.strptime(row[0], "%Y-%m-%d")
                    exit_date_obj = datetime.strptime(exit_date, "%Y-%m-%d")
                    hold_days = (exit_date_obj - entry_date).days
                except:
                    pass
            
            cursor.execute('''
                UPDATE recommendations 
                SET close_price = ?, 
                    pnl = ?, 
                    status = 'CLOSED',
                    exit_date = ?,
                    exit_price = ?,
                    hold_days = ?,
                    exit_reason = ?,
                    max_profit = ?,
                    max_loss = ?,
                    relative_return = ?
                WHERE id = ?
            ''', (
                exit_price, pnl, exit_date, exit_price, hold_days,
                exit_reason, max_profit, max_loss, relative_return, rec_id
            ))
            conn.commit()
        finally:
            conn.close()

    def get_weights(self):
        conn = self.get_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM strategy_params", conn)
            # 转换为字典
            return dict(zip(df['key'], df['value']))
        finally:
            conn.close()

    def update_weight(self, key, value):
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE strategy_params 
                SET value = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE key = ?
            ''', (value, key))
            conn.commit()
        finally:
            conn.close()

    def save_daily_scan(self, date_str: str, results: list):
        """保存每日 Top 扫描快照（使用压缩存储）"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 转换为JSON字符串
            json_str = json.dumps(results, ensure_ascii=False)
            # 使用gzip压缩
            compressed_data = gzip.compress(json_str.encode('utf-8'), compresslevel=6)
            cursor.execute('''
                INSERT OR REPLACE INTO daily_scans (scan_date, results_json)
                VALUES (?, ?)
            ''', (date_str, compressed_data))
            conn.commit()
        finally:
            conn.close()

    def get_daily_scan(self, date_str: str):
        """获取指定日期的扫描快照（支持压缩存储）"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT results_json FROM daily_scans WHERE scan_date = ?", (date_str,))
            row = cursor.fetchone()
            if row:
                data = row[0]
                # 尝试解压（如果是压缩数据）
                try:
                    if isinstance(data, bytes):
                        json_str = gzip.decompress(data).decode('utf-8')
                    else:
                        # 兼容旧格式（未压缩的JSON字符串）
                        try:
                            json_str = gzip.decompress(data.encode('utf-8')).decode('utf-8')
                        except:
                            json_str = data
                    return json.loads(json_str)
                except Exception as e:
                    # 如果解压失败，尝试直接解析（兼容旧数据）
                    try:
                        return json.loads(data)
                    except:
                        print(f"[Database] Error decompressing daily scan data for {date_str}: {e}")
                        return None
            return None
        finally:
            conn.close()

    def get_cached_kline(self, symbol: str):
        """获取本地缓存的 K 线数据（支持压缩存储）"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM kline_cache WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if row:
                data = row[0]
                # 尝试解压（如果是压缩数据）
                try:
                    # 如果是字节类型，尝试解压
                    if isinstance(data, bytes):
                        json_str = gzip.decompress(data).decode('utf-8')
                    else:
                        # 如果是字符串，可能是旧格式（未压缩）或已经是JSON字符串
                        # 尝试解压，如果失败则直接使用
                        try:
                            json_str = gzip.decompress(data.encode('utf-8')).decode('utf-8')
                        except:
                            json_str = data
                    return pd.read_json(StringIO(json_str))
                except Exception as e:
                    # 如果解压失败，尝试直接作为JSON解析（兼容旧数据）
                    try:
                        return pd.read_json(StringIO(data))
                    except:
                        print(f"[Database] Error decompressing kline data for {symbol}: {e}")
                        return None
            return None
        finally:
            conn.close()

    def save_kline(self, symbol: str, df: pd.DataFrame):
        """缓存 K 线数据到本地（使用压缩存储）"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 转换为JSON字符串
            json_str = df.to_json(orient='records', date_format='iso')
            # 使用gzip压缩（compresslevel=6是速度和压缩率的平衡）
            compressed_data = gzip.compress(json_str.encode('utf-8'), compresslevel=6)
            # 存储压缩后的数据（SQLite会自动处理BLOB）
            cursor.execute("INSERT OR REPLACE INTO kline_cache (symbol, data_json) VALUES (?, ?)", 
                           (symbol, compressed_data))
            conn.commit()
        finally:
            conn.close()

    def get_cached_stock_info(self, symbol: str):
        """获取本地缓存的个股画像"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM stock_info_cache WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
        finally:
            conn.close()

    def save_stock_info(self, symbol: str, data: dict):
        """保存个股画像到本地"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO stock_info_cache (symbol, data_json) VALUES (?, ?)", 
                           (symbol, json.dumps(data)))
            conn.commit()
        finally:
            conn.close()

    def get_performance_stats(self):
        conn = self.get_connection()
        try:
            return pd.read_sql_query("SELECT * FROM recommendations WHERE status = 'CLOSED'", conn)
        finally:
            conn.close()

    def save_news_batch(self, news_list: list):
        """批量保存已分析的新闻，并更新行业情绪值"""
        if not news_list:
            return
            
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 统一使用北京时间 (UTC+8)
            beijing_now = datetime.utcnow() + timedelta(hours=8)
            current_date = beijing_now.date()
            current_year = beijing_now.year
            week_start = self.get_week_start_date()
                    
            # 用于累积本批次的情绪值，最后统一更新数据库，减少 IO
            sentiment_updates = {} # {industry: {'score_sum': float, 'count': int}}
        
            import hashlib
            for item in news_list:
                # 生成唯一主键：标题 + 发布时间
                raw_time = item.get('time', '')
                if raw_time is None: raw_time = ""
                publish_time_str = str(raw_time).strip()
                        
                # 智能补全日期格式
                if publish_time_str:
                    # 情况 1: 只有时间 "HH:MM:SS" 或 "HH:MM"
                    if ':' in publish_time_str and len(publish_time_str) <= 8 and '-' not in publish_time_str:
                        publish_time_str = f"{current_date.strftime('%Y-%m-%d')} {publish_time_str}"
                            
                    # 情况 2: 只有月日 "MM-DD HH:MM"
                    elif publish_time_str.count('-') == 1 and publish_time_str.find('-') < 5:
                        publish_time_str = f"{current_year}-{publish_time_str}"
                            
                    # 情况 3: 只有月日但用斜杠 "MM/DD HH:MM"
                    elif publish_time_str.count('/') == 1 and publish_time_str.find('/') < 5:
                        publish_time_str = f"{current_year}-{publish_time_str.replace('/', '-')}"
                        
                unique_str = f"{item.get('title', '')}_{publish_time_str}"
                # 使用 md5 生成稳定的哈希值，避免 Python hash() 随进程重启而变化
                title_hash = hashlib.md5(unique_str.encode('utf-8')).hexdigest() 
                            
                # 检查 sentiment 是否为字典，需序列化
                sentiment_data = item.get('sentiment', {})
                if isinstance(sentiment_data, dict):
                    sentiment_json = json.dumps(sentiment_data)
                    sector = sentiment_data.get('sector')
                    score = sentiment_data.get('score', 0.0)
                else:
                    sentiment_json = str(sentiment_data)
                    sector = None
                    score = 0.0
    
                # 尝试插入新闻
                cursor.execute('''
                    INSERT OR IGNORE INTO news_analysis (title_hash, title, content, publish_time, sentiment_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    title_hash, 
                    item.get('title', ''), 
                    item.get('content', ''), 
                    publish_time_str, 
                    sentiment_json
                ))
                    
                # 如果插入成功（rowcount > 0），则说明是新新闻，需要累积情绪值
                if cursor.rowcount > 0 and sector:
                    # 记录行业情绪
                    if sector not in sentiment_updates:
                        sentiment_updates[sector] = {'score_sum': 0.0, 'count': 0}
                    sentiment_updates[sector]['score_sum'] += float(score)
                    sentiment_updates[sector]['count'] += 1
                        
                    # 记录全市场情绪
                    if "全市场" not in sentiment_updates:
                        sentiment_updates["全市场"] = {'score_sum': 0.0, 'count': 0}
                    sentiment_updates["全市场"]["score_sum"] += float(score)
                    sentiment_updates["全市场"]["count"] += 1
                
            # 批量更新行业情绪值表
            for industry, data in sentiment_updates.items():
                # 检查是否存在记录
                cursor.execute('''
                    SELECT sentiment_score, news_count 
                    FROM industry_sentiment_weekly 
                    WHERE industry = ? AND week_start_date = ?
                ''', (industry, week_start))
                    
                row = cursor.fetchone()
                if row:
                    # 累积更新：基础分 + (总分 * 0.1)
                    # 注意：sentiment_score 存储的是 1.0 + sum(individual_score * 0.1)
                    new_score = row[0] + (data['score_sum'] * 0.1)
                    new_count = row[1] + data['count']
                    cursor.execute('''
                        UPDATE industry_sentiment_weekly 
                        SET sentiment_score = ?, news_count = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE industry = ? AND week_start_date = ?
                    ''', (new_score, new_count, industry, week_start))
                else:
                    # 新建记录：初始分 1.0 + (总分 * 0.1)
                    cursor.execute('''
                        INSERT INTO industry_sentiment_weekly (industry, week_start_date, sentiment_score, news_count)
                        VALUES (?, ?, ?, ?)
                    ''', (industry, week_start, 1.0 + (data['score_sum'] * 0.1), data['count']))
                
            conn.commit()
        except Exception as e:
            print(f"[Database] Error saving news batch: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_latest_news(self, limit: int = 20, max_age_days: int = 3):
        """
        获取最新已分析的新闻（增强排序稳定性）
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 统一使用北京时间计算截止日期
            beijing_now = datetime.utcnow() + timedelta(hours=8)
            cutoff_date = beijing_now - timedelta(days=max_age_days)
            
            # 先按创建时间取最近的，保证数据范围正确
            cursor.execute("""
                SELECT title, content, publish_time, sentiment_json, created_at 
                FROM news_analysis 
                WHERE created_at >= ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, (cutoff_date, limit))
            rows = cursor.fetchall()
            
            result = []
            current_year = beijing_now.year
            for row in rows:
                try:
                    sentiment = json.loads(row[3])
                except:
                    sentiment = {}
                
                publish_time = str(row[2] or "").strip()
                created_at = row[4]
                
                # 归一化处理：确保所有返回的时间都带有正确的年份 YYYY-MM-DD
                if publish_time:
                    # 如果没有年份 (MM-DD 或 HH:MM)
                    if publish_time.count('-') == 1 and publish_time.find('-') < 5:
                        publish_time = f"{current_year}-{publish_time}"
                    elif ':' in publish_time and '-' not in publish_time:
                        # 只有时间，补全创建时的日期
                        try:
                            c_date = datetime.strptime(created_at.split()[0], "%Y-%m-%d")
                            publish_time = f"{c_date.strftime('%Y-%m-%d')} {publish_time}"
                        except:
                            publish_time = f"{beijing_now.strftime('%Y-%m-%d')} {publish_time}"
                
                result.append({
                    "title": row[0],
                    "content": row[1],
                    "time": publish_time,
                    "sentiment": sentiment,
                    "created_at": created_at
                })
            
            # 在返回前，进行一次基于规范化日期的时间戳排序
            def parse_to_ts(item):
                try:
                    t_str = item['time']
                    if len(t_str) <= 10: t_str += " 00:00:00"
                    return datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").timestamp()
                except:
                    return 0

            result.sort(key=parse_to_ts, reverse=True)
            return result
        finally:
            conn.close()

    def log_trade(self, trade_data: dict):
        """记录一笔新的模拟交易"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO one_night_trades (
                    symbol, name, buy_date, buy_price, quantity, amount, fees, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'HOLD')
            ''', (
                trade_data['symbol'], trade_data['name'], trade_data['buy_date'],
                trade_data['buy_price'], trade_data['quantity'], trade_data['amount'],
                trade_data.get('fees', 0.0)
            ))
            conn.commit()
        finally:
            conn.close()

    def close_trade(self, trade_id: int, sell_data: dict):
        """平仓交易"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE one_night_trades 
                SET sell_date = ?, sell_price = ?, sell_amount = ?, 
                    pnl = ?, pnl_pct = ?, fees = fees + ?, status = 'SOLD'
                WHERE id = ?
            ''', (
                sell_data['sell_date'], sell_data['sell_price'], sell_data['sell_amount'],
                sell_data['pnl'], sell_data['pnl_pct'], sell_data.get('sell_fees', 0.0),
                trade_id
            ))
            conn.commit()
        finally:
            conn.close()

    def get_active_trades(self):
        """获取当前持仓"""
        conn = self.get_connection()
        try:
            return pd.read_sql_query("SELECT * FROM one_night_trades WHERE status = 'HOLD'", conn)
        finally:
            conn.close()

    def get_trade_history(self, limit: int = 50):
        """获取历史交易记录"""
        conn = self.get_connection()
        try:
            return pd.read_sql_query(f"SELECT * FROM one_night_trades ORDER BY created_at DESC LIMIT {limit}", conn)
        finally:
            conn.close()
    
    def get_week_start_date(self, date_obj=None):
        """获取指定日期所在周的开始日期（周一） - 使用北京时间"""
        if date_obj is None:
            # 统一使用北京时间
            date_obj = (datetime.utcnow() + timedelta(hours=8)).date()
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()
        
        # 获取周一（weekday() 返回 0-6，0是周一）
        days_since_monday = date_obj.weekday()
        week_start = date_obj - timedelta(days=days_since_monday)
        return week_start.strftime("%Y-%m-%d")
    
    def update_industry_sentiment(self, industry: str, sentiment_score: float):
        """
        更新行业情绪值（累积模式）
        
        Args:
            industry: 行业名称（如 "半导体"、"全市场"）
            sentiment_score: 新闻的情感得分（-1.0 到 1.0）
        """
        if not industry:
            return
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            week_start = self.get_week_start_date()
            
            # 获取当前周的情绪值，如果不存在则创建（默认1.0）
            cursor.execute('''
                SELECT sentiment_score, news_count 
                FROM industry_sentiment_weekly 
                WHERE industry = ? AND week_start_date = ?
            ''', (industry, week_start))
            
            row = cursor.fetchone()
            if row:
                current_score = row[0]
                news_count = row[1]
            else:
                # 新周开始，初始化为1.0
                current_score = 1.0
                news_count = 0
                cursor.execute('''
                    INSERT INTO industry_sentiment_weekly (industry, week_start_date, sentiment_score, news_count)
                    VALUES (?, ?, ?, ?)
                ''', (industry, week_start, current_score, news_count))
            
            # 累积情绪值：基础分1.0 + 新闻得分
            # 使用加权平均，避免单条新闻影响过大
            new_score = current_score + (sentiment_score * 0.1)  # 每条新闻影响0.1分
            new_news_count = news_count + 1
            
            # 更新情绪值和新闻数量
            cursor.execute('''
                UPDATE industry_sentiment_weekly 
                SET sentiment_score = ?, news_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE industry = ? AND week_start_date = ?
            ''', (new_score, new_news_count, industry, week_start))
            
            conn.commit()
        except Exception as e:
            print(f"[Database] Error updating industry sentiment: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_industry_sentiment_weekly(self, week_start_date: str = None):
        """
        获取指定周的各行业情绪值
        
        Args:
            week_start_date: 周开始日期（YYYY-MM-DD），如果为None则返回当前周
        
        Returns:
            List[Dict]: 行业情绪值列表，每个元素包含 industry, sentiment_score, news_count
        """
        if week_start_date is None:
            week_start_date = self.get_week_start_date()
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT industry, sentiment_score, news_count
                FROM industry_sentiment_weekly
                WHERE week_start_date = ?
                ORDER BY sentiment_score DESC
            ''', (week_start_date,))
            
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "industry": row[0],
                    "sentiment_score": row[1],
                    "news_count": row[2]
                })
            return result
        finally:
            conn.close()
    
    def get_current_and_last_week_sentiment(self):
        """
        获取当前周和上周的行业情绪值
        
        Returns:
            Dict: {
                "current_week": List[Dict],
                "last_week": List[Dict],
                "current_week_start": str,
                "last_week_start": str,
                "market_sentiment": float  # 全市场情绪值
            }
        """
        current_week_start = self.get_week_start_date()
        current_date = datetime.now().date()
        # 计算上周开始日期
        days_since_monday = current_date.weekday()
        last_week_start = (current_date - timedelta(days=days_since_monday + 7)).strftime("%Y-%m-%d")
        
        current_week_data = self.get_industry_sentiment_weekly(current_week_start)
        last_week_data = self.get_industry_sentiment_weekly(last_week_start)
        
        # 提取全市场情绪值
        market_sentiment = 1.0
        for item in current_week_data:
            if item["industry"] == "全市场":
                market_sentiment = item["sentiment_score"]
                break
        
        return {
            "current_week": current_week_data,
            "last_week": last_week_data,
            "current_week_start": current_week_start,
            "last_week_start": last_week_start,
            "market_sentiment": market_sentiment
        }
    
    def initialize_industry_sentiment_from_history(self):
        """
        从历史新闻数据初始化行业情绪值（用于首次使用或数据迁移）
        查询一周内的所有新闻，根据行业和情感得分累积计算各行业的情绪值
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # 获取当前周的开始日期
            week_start = self.get_week_start_date()
            
            # 检查当前周是否已有数据
            cursor.execute('''
                SELECT COUNT(*) FROM industry_sentiment_weekly 
                WHERE week_start_date = ?
            ''', (week_start,))
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                print(f"[Database] Industry sentiment for week {week_start} already exists. Skipping initialization.")
                return existing_count
            
            # 查询一周内的所有新闻（从本周一开始到今天）
            current_date = datetime.now().date()
            days_since_monday = current_date.weekday()
            week_start_date_obj = current_date - timedelta(days=days_since_monday)
            
            # 查询本周内的所有新闻（使用日期字符串格式匹配）
            week_start_str = week_start_date_obj.strftime("%Y-%m-%d")
            print(f"[Database] Querying news from {week_start_str} (week start: {week_start})")
            # SQLite 的日期比较：created_at 是 TIMESTAMP，需要转换为日期字符串比较
            cursor.execute('''
                SELECT sentiment_json, created_at
                FROM news_analysis
                WHERE DATE(created_at) >= ?
                ORDER BY created_at ASC
            ''', (week_start_str,))
            
            rows = cursor.fetchall()
            print(f"[Database] Found {len(rows)} news items for week {week_start}")
            
            if not rows:
                print(f"[Database] No news found for week {week_start}. Cannot initialize sentiment.")
                return 0
            
            # 统计各行业的情绪值
            industry_scores = {}  # {industry: {'score': float, 'count': int}}
            
            for row in rows:
                try:
                    sentiment_json = row[0]
                    if not sentiment_json:
                        continue
                    
                    sentiment = json.loads(sentiment_json)
                    if not isinstance(sentiment, dict):
                        continue
                    
                    sector = sentiment.get('sector', '')
                    score = sentiment.get('score', 0.0)
                    
                    if not sector or not isinstance(score, (int, float)):
                        continue
                    
                    # 累积各行业的情绪值
                    if sector not in industry_scores:
                        industry_scores[sector] = {'score': 1.0, 'count': 0}  # 基础分1.0
                    
                    # 每条新闻影响0.1分
                    industry_scores[sector]['score'] += float(score) * 0.1
                    industry_scores[sector]['count'] += 1
                    
                except Exception as e:
                    print(f"[Database] Error processing news sentiment: {e}")
                    continue
            
            # 将统计结果写入数据库
            inserted_count = 0
            for industry, data in industry_scores.items():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO industry_sentiment_weekly 
                        (industry, week_start_date, sentiment_score, news_count, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (industry, week_start, data['score'], data['count']))
                    inserted_count += 1
                except Exception as e:
                    print(f"[Database] Error inserting industry sentiment for {industry}: {e}")
            
            conn.commit()
            print(f"[Database] Initialized {inserted_count} industry sentiment records from {len(rows)} historical news items.")
            return inserted_count
            
        except Exception as e:
            print(f"[Database] Error initializing industry sentiment from history: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def cleanup_old_news(self, max_age_days: int = 7):
        """
        清理过期的新闻（定期清理，释放存储空间）
        
        Args:
            max_age_days: 保留天数，超过此天数的新闻将被删除（默认7天）
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cursor.execute("DELETE FROM news_analysis WHERE created_at < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            print(f"[Database] Cleaned up {deleted_count} old news items (older than {max_age_days} days).")
            return deleted_count
        except Exception as e:
            print(f"[Database] Error cleaning up old news: {e}")
            return 0
        finally:
            conn.close()

if __name__ == "__main__":
    db = DatabaseManager()
    print("Database initialized at", DB_PATH)
