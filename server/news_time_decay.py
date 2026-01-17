"""
新闻时效性处理模块：实现时间衰减和过期清理
"""
import datetime
from typing import List, Dict
import json

class NewsTimeDecay:
    """新闻时效性处理：时间衰减和过期管理"""
    
    def __init__(self, max_age_days: int = 3, trading_days_only: bool = True):
        """
        初始化时效性处理器
        
        Args:
            max_age_days: 新闻最大保留天数（默认3天）
            trading_days_only: 是否只考虑交易日（默认True，周末的新闻在周一仍然有效）
        """
        self.max_age_days = max_age_days
        self.trading_days_only = trading_days_only
    
    def is_trading_day(self, date: datetime.date) -> bool:
        """判断是否为交易日（简单实现：排除周末）"""
        # 0 = Monday, 6 = Sunday
        return date.weekday() < 5
    
    def get_trading_days_between(self, start_date: datetime.date, end_date: datetime.date) -> int:
        """计算两个日期之间的交易日数量"""
        if not self.trading_days_only:
            return (end_date - start_date).days
        
        trading_days = 0
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days += 1
            current += datetime.timedelta(days=1)
        return trading_days
    
    def calculate_time_decay_weight(
        self, 
        publish_time: str, 
        current_time: datetime.datetime = None
    ) -> float:
        """
        计算新闻的时间衰减权重
        
        Args:
            publish_time: 新闻发布时间（字符串格式，如 "14:30:00" 或 "2024-01-15 14:30:00"）
            current_time: 当前时间（默认使用当前时间）
            
        Returns:
            权重值 (0.0 到 1.0)，1.0 表示最新，0.0 表示已过期
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        
        # 尝试解析发布时间
        try:
            # 如果只有时间（HH:MM:SS），假设是今天
            if ':' in publish_time and len(publish_time.split(':')) == 3:
                time_parts = publish_time.split(':')
                if len(time_parts) == 3 and len(publish_time) <= 8:
                    # 只有时间，没有日期，假设是今天
                    today = current_time.date()
                    publish_dt = datetime.datetime.combine(
                        today, 
                        datetime.time(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
                    )
                else:
                    # 尝试解析完整日期时间
                    publish_dt = datetime.datetime.strptime(publish_time, "%Y-%m-%d %H:%M:%S")
            else:
                # 尝试其他格式
                publish_dt = datetime.datetime.strptime(publish_time, "%Y-%m-%d %H:%M:%S")
        except:
            # 解析失败，返回默认权重
            return 0.5
        
        # 计算时间差
        time_diff = current_time - publish_dt
        
        # 如果超过最大保留天数，权重为0
        if time_diff.days > self.max_age_days:
            return 0.0
        
        # 计算交易日差（如果启用交易日模式）
        if self.trading_days_only:
            publish_date = publish_dt.date()
            current_date = current_time.date()
            trading_days_diff = self.get_trading_days_between(publish_date, current_date)
            
            if trading_days_diff > self.max_age_days:
                return 0.0
            
            # 使用指数衰减：权重 = e^(-k * trading_days)
            # k 值使得 max_age_days 时权重约为 0.1
            k = 0.5 / self.max_age_days
            weight = max(0.0, min(1.0, 1.0 - (trading_days_diff * k)))
        else:
            # 使用线性衰减
            weight = max(0.0, 1.0 - (time_diff.days / self.max_age_days))
        
        return weight
    
    def filter_news_by_age(
        self, 
        news_list: List[Dict], 
        current_time: datetime.datetime = None
    ) -> List[Dict]:
        """
        根据时效性过滤新闻列表
        
        Args:
            news_list: 新闻列表，每个新闻包含 'time' 字段
            current_time: 当前时间（默认使用当前时间）
            
        Returns:
            过滤后的新闻列表（只包含未过期的新闻）
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        
        filtered_news = []
        for news in news_list:
            publish_time = news.get('time', '')
            if not publish_time:
                continue
            
            weight = self.calculate_time_decay_weight(publish_time, current_time)
            
            # 只保留权重 > 0 的新闻（未过期）
            if weight > 0:
                # 添加时间衰减权重到新闻数据中
                news_with_weight = news.copy()
                news_with_weight['time_decay_weight'] = weight
                filtered_news.append(news_with_weight)
        
        return filtered_news
    
    def apply_time_decay_to_sentiment_score(
        self, 
        base_score: float, 
        time_decay_weight: float
    ) -> float:
        """
        将时间衰减权重应用到情感得分上
        
        Args:
            base_score: 原始情感得分 (-1.0 到 1.0)
            time_decay_weight: 时间衰减权重 (0.0 到 1.0)
            
        Returns:
            调整后的情感得分
        """
        return base_score * time_decay_weight

if __name__ == "__main__":
    # 测试代码
    decay = NewsTimeDecay(max_age_days=3, trading_days_only=True)
    
    # 测试时间衰减权重计算
    print("测试时间衰减权重：")
    test_times = [
        "14:30:00",  # 今天
        "09:00:00",  # 今天
        "15:00:00",  # 今天
    ]
    
    for time_str in test_times:
        weight = decay.calculate_time_decay_weight(time_str)
        print(f"  时间: {time_str}, 权重: {weight:.3f}")
    
    # 测试新闻过滤
    print("\n测试新闻过滤：")
    test_news = [
        {"title": "最新新闻", "content": "内容1", "time": "14:30:00"},
        {"title": "旧新闻", "content": "内容2", "time": "09:00:00"},
    ]
    
    filtered = decay.filter_news_by_age(test_news)
    print(f"  原始新闻数: {len(test_news)}")
    print(f"  过滤后新闻数: {len(filtered)}")
    for news in filtered:
        print(f"    - {news['title']}: 权重={news.get('time_decay_weight', 0):.3f}")





