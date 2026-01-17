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
        conn = sqlite3.connect(DB_PATH, timeout=30) # 增加超时时间防止锁死
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
        """批量保存已分析的新闻"""
        if not news_list:
            return
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            current_date = datetime.now().date()
            for item in news_list:
                # 生成唯一主键：标题 + 发布时间
                raw_time = item.get('time', '')
                if raw_time is None: raw_time = ""
                # 强转为字符串，防止 datetime.time 类型导致 sqlite 报错
                publish_time_str = str(raw_time)
                
                # 如果时间只有时间没有日期（格式如 "14:30:00"），添加当前日期
                if ':' in publish_time_str and len(publish_time_str.split(':')) >= 2:
                    # 检查是否只有时间（长度 <= 8，如 "14:30:00"）
                    if len(publish_time_str) <= 8 and '-' not in publish_time_str:
                        # 只有时间，添加当前日期
                        publish_time_str = f"{current_date.strftime('%Y-%m-%d')} {publish_time_str}"
                    # 如果已经是完整日期时间格式，保持不变
                
                unique_str = f"{item.get('title', '')}_{publish_time_str}"
                title_hash = str(hash(unique_str)) 
                
                # 检查 sentiment 是否为字典，需序列化
                sentiment_data = item.get('sentiment', {})
                if isinstance(sentiment_data, dict):
                    sentiment_json = json.dumps(sentiment_data)
                else:
                    sentiment_json = str(sentiment_data)

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
            conn.commit()
        except Exception as e:
            print(f"[Database] Error saving news batch: {e}")
        finally:
            conn.close()

    def get_latest_news(self, limit: int = 20, max_age_days: int = 3):
        """
        获取最新已分析的新闻（按创建时间降序，最新的在最前面）
        
        Args:
            limit: 返回的最大新闻数量
            max_age_days: 新闻最大保留天数（默认3天），超过此天数的新闻将被过滤
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            # 使用 created_at 排序，因为它是 TIMESTAMP 类型，可以正确按时间排序
            # 如果 publish_time 包含完整日期时间，也可以尝试使用它，但为了保险起见使用 created_at
            # 添加时间过滤：只获取最近 max_age_days 天的新闻
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cursor.execute("""
                SELECT title, content, publish_time, sentiment_json, created_at 
                FROM news_analysis 
                WHERE created_at >= ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, (cutoff_date, limit))
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                try:
                    sentiment = json.loads(row[3])
                except:
                    sentiment = {}
                
                # 确保时间格式包含日期（如果只有时间，使用 created_at 的日期）
                publish_time = row[2] or ""
                created_at = row[4]
                
                # 如果 publish_time 只有时间没有日期，使用 created_at 的日期
                if publish_time:
                    publish_time_str = str(publish_time).strip()
                    # 检查是否只有时间（格式如 "19:34:12" 或 "19:34"）
                    if ':' in publish_time_str and len(publish_time_str.split(':')) >= 2:
                        # 检查是否包含日期（包含 '-' 且长度 > 10）
                        if len(publish_time_str) <= 8 or '-' not in publish_time_str:
                            # 只有时间，使用 created_at 的日期
                            if created_at:
                                try:
                                    if isinstance(created_at, str):
                                        # 尝试多种日期格式
                                        try:
                                            created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                                        except:
                                            try:
                                                created_dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S.%f")
                                            except:
                                                created_dt = datetime.now()
                                    else:
                                        created_dt = created_at
                                    publish_time = f"{created_dt.strftime('%Y-%m-%d')} {publish_time_str}"
                                except Exception as e:
                                    # 如果解析失败，使用当前日期
                                    publish_time = f"{datetime.now().strftime('%Y-%m-%d')} {publish_time_str}"
                            else:
                                # 没有 created_at，使用当前日期
                                publish_time = f"{datetime.now().strftime('%Y-%m-%d')} {publish_time_str}"
                
                result.append({
                    "title": row[0],
                    "content": row[1],
                    "time": publish_time,
                    "sentiment": sentiment,
                    "created_at": row[4]  # 添加创建时间，用于进一步的时间衰减计算
                })
            return result
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
