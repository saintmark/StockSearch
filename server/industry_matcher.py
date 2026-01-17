"""
行业匹配模块：将LLM分析的行业与股票的实际行业进行关联匹配
"""

from typing import Dict, List, Optional
import re

class IndustryMatcher:
    """行业匹配器：支持模糊匹配和关键词映射"""
    
    def __init__(self):
        # LLM分析的行业 -> 股票实际行业的映射表
        # 支持一对多映射（一个LLM行业可能对应多个股票行业）
        self.llm_to_stock_industry_map = {
            '半导体/芯片': [
                '半导体', '芯片', '集成电路', '电子元件', '电子制造', 
                '光学光电子', '电子化学品', '印制电路板', '被动元件'
            ],
            '人工智能': [
                '软件开发', '计算机应用', 'IT服务', '互联网服务', 
                '通信设备', '通信服务', '计算机设备', '人工智能'
            ],
            '新能源': [
                '新能源', '光伏设备', '风电设备', '电池', '储能', 
                '电力设备', '新能源车', '充电桩', '锂电池', '太阳能'
            ],
            '金融/地产': [
                '银行', '证券', '保险', '房地产', '房地产开发', 
                '房地产服务', '多元金融', '信托', '租赁'
            ],
            '大消费': [
                '白酒', '食品饮料', '餐饮', '零售', '商贸', 
                '消费电子', '家用电器', '纺织服装', '轻工制造', '汽车'
            ],
            '宏观经济': [
                '全市场'  # 宏观经济影响全市场
            ],
            '全市场': [
                '全市场'  # 全市场新闻影响所有行业
            ]
        }
        
        # 反向映射：股票行业 -> LLM行业的映射（用于快速查找）
        self._build_reverse_map()
        
        # 行业关键词匹配规则（用于模糊匹配）
        self.industry_keywords = {
            '半导体': ['半导体', '芯片', '集成电路', 'IC', '晶圆', '封测', '光刻'],
            '新能源': ['新能源', '光伏', '风电', '电池', '储能', '锂电', '太阳能', '充电桩'],
            '人工智能': ['AI', '人工智能', '大模型', '算力', '机器学习', '深度学习', 'ChatGPT'],
            '金融': ['银行', '证券', '保险', '金融', '信托', '基金'],
            '地产': ['房地产', '地产', '开发', '物业', '建筑'],
            '消费': ['消费', '白酒', '食品', '饮料', '零售', '餐饮', '家电'],
            '汽车': ['汽车', '新能源车', '电动车', '整车', '零部件'],
            '医药': ['医药', '生物', '医疗', '制药', '疫苗', '医疗器械'],
            '通信': ['通信', '5G', '6G', '网络', '电信', '移动'],
            '化工': ['化工', '化学', '材料', '塑料', '橡胶'],
            '钢铁': ['钢铁', '金属', '有色金属', '铝', '铜'],
            '煤炭': ['煤炭', '采掘', '石油', '天然气'],
            '电力': ['电力', '发电', '电网', '能源'],
            '交通运输': ['交通', '物流', '运输', '港口', '航空', '铁路'],
            '建筑': ['建筑', '工程', '装饰', '建材'],
            '传媒': ['传媒', '文化', '娱乐', '影视', '游戏', '广告'],
            '军工': ['军工', '国防', '航空装备', '船舶'],
        }
    
    def _build_reverse_map(self):
        """构建反向映射表"""
        self.stock_to_llm_map = {}
        for llm_industry, stock_industries in self.llm_to_stock_industry_map.items():
            for stock_industry in stock_industries:
                if stock_industry not in self.stock_to_llm_map:
                    self.stock_to_llm_map[stock_industry] = []
                self.stock_to_llm_map[stock_industry].append(llm_industry)
    
    def match_industry(self, llm_sector: str, stock_industry: str) -> bool:
        """
        判断LLM分析的行业是否与股票行业匹配
        
        Args:
            llm_sector: LLM分析得出的行业（如"半导体/芯片"）
            stock_industry: 股票的实际行业（如"半导体"）
            
        Returns:
            bool: 是否匹配
        """
        if not llm_sector or not stock_industry:
            return False
        
        # 1. 精确匹配：检查映射表
        if llm_sector in self.llm_to_stock_industry_map:
            mapped_industries = self.llm_to_stock_industry_map[llm_sector]
            # 检查股票行业是否在映射列表中
            for mapped in mapped_industries:
                if mapped in stock_industry or stock_industry in mapped:
                    return True
        
        # 2. 模糊匹配：使用关键词
        if stock_industry in self.stock_to_llm_map:
            mapped_llm_sectors = self.stock_to_llm_map[stock_industry]
            if llm_sector in mapped_llm_sectors:
                return True
        
        # 3. 关键词模糊匹配
        for keyword, industries in self.industry_keywords.items():
            # 检查LLM行业是否包含关键词
            llm_has_keyword = any(kw in llm_sector for kw in industries)
            # 检查股票行业是否包含关键词
            stock_has_keyword = any(kw in stock_industry for kw in industries)
            
            if llm_has_keyword and stock_has_keyword:
                return True
        
        # 4. 直接字符串包含匹配（兜底）
        if llm_sector in stock_industry or stock_industry in llm_sector:
            return True
        
        # 5. 全市场新闻匹配所有行业
        if llm_sector in ['全市场', '宏观经济']:
            return True
        
        return False
    
    def get_matching_news(
        self, 
        news_list: List[Dict], 
        stock_industry: str,
        stock_name: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        从新闻列表中筛选出与股票相关的新闻
        
        Args:
            news_list: 新闻列表，每个新闻包含 sentiment 字段（包含 sector）
            stock_industry: 股票行业
            stock_name: 股票名称（可选）
            symbol: 股票代码（可选）
            
        Returns:
            匹配的新闻列表
        """
        matching_news = []
        
        for news in news_list:
            # 获取新闻的行业分析结果
            sentiment = news.get('sentiment', {})
            if isinstance(sentiment, str):
                try:
                    import json
                    sentiment = json.loads(sentiment)
                except:
                    sentiment = {}
            
            llm_sector = sentiment.get('sector', '')
            
            # 检查1: 股票名称或代码是否在新闻内容中出现（直接提及）
            content = news.get('content', '') or news.get('内容', '')
            title = news.get('title', '') or news.get('标题', '')
            full_text = f"{title} {content}"
            
            is_direct_mention = False
            if stock_name and stock_name in full_text:
                is_direct_mention = True
            if symbol and symbol in full_text:
                is_direct_mention = True
            
            # 检查2: 行业是否匹配
            is_industry_match = self.match_industry(llm_sector, stock_industry)
            
            # 如果直接提及或行业匹配，则加入匹配列表
            if is_direct_mention or is_industry_match:
                news_copy = news.copy()
                news_copy['match_type'] = 'direct' if is_direct_mention else 'industry'
                news_copy['llm_sector'] = llm_sector
                matching_news.append(news_copy)
        
        return matching_news
    
    def calculate_sentiment_score(
        self, 
        matching_news: List[Dict],
        direct_mention_weight: float = 0.3,
        industry_match_weight: float = 0.15,
        time_decay_handler = None
    ) -> float:
        """
        根据匹配的新闻计算情感得分（支持时间衰减）
        
        Args:
            matching_news: 匹配的新闻列表
            direct_mention_weight: 直接提及的权重（每条）
            industry_match_weight: 行业匹配的权重（每条）
            time_decay_handler: 时间衰减处理器（NewsTimeDecay 实例），如果提供则应用时间衰减
            
        Returns:
            归一化的情感得分 (-1 到 1)
        """
        if not matching_news:
            return 0.0
        
        total_score = 0.0
        direct_count = 0
        industry_count = 0
        
        for news in matching_news:
            sentiment = news.get('sentiment', {})
            if isinstance(sentiment, str):
                try:
                    import json
                    sentiment = json.loads(sentiment)
                except:
                    sentiment = {}
            
            # 获取新闻的情感得分
            news_score = sentiment.get('score', 0.0)
            if not isinstance(news_score, (int, float)):
                news_score = 0.0
            
            # 应用时间衰减（如果提供了时间衰减处理器）
            time_decay_weight = news.get('time_decay_weight', 1.0)
            if time_decay_handler and 'time' in news:
                # 如果新闻还没有计算时间衰减权重，则计算
                if 'time_decay_weight' not in news:
                    time_decay_weight = time_decay_handler.calculate_time_decay_weight(news.get('time', ''))
                else:
                    time_decay_weight = news.get('time_decay_weight', 1.0)
            
            # 应用时间衰减到新闻得分
            news_score = news_score * time_decay_weight
            
            match_type = news.get('match_type', 'industry')
            
            if match_type == 'direct':
                # 直接提及：权重更高
                total_score += news_score * direct_mention_weight
                direct_count += 1
            else:
                # 行业匹配：权重较低
                total_score += news_score * industry_match_weight
                industry_count += 1
        
        # 归一化到 -1 到 1 范围
        # 考虑新闻数量，避免单条新闻影响过大
        normalized_score = max(-1.0, min(1.0, total_score))
        
        return normalized_score

if __name__ == "__main__":
    # 测试代码
    matcher = IndustryMatcher()
    
    # 测试1: 精确匹配
    print("测试1: 半导体行业匹配")
    print(matcher.match_industry("半导体/芯片", "半导体"))  # 应该返回 True
    
    # 测试2: 模糊匹配
    print("\n测试2: 新能源行业匹配")
    print(matcher.match_industry("新能源", "光伏设备"))  # 应该返回 True
    
    # 测试3: 新闻匹配
    print("\n测试3: 新闻匹配测试")
    test_news = [
        {
            'content': '半导体行业迎来重大突破',
            'title': '芯片技术突破',
            'sentiment': {'sector': '半导体/芯片', 'score': 0.8, 'sentiment': 'positive'}
        },
        {
            'content': '新能源车销量大增',
            'title': '新能源市场火爆',
            'sentiment': {'sector': '新能源', 'score': 0.6, 'sentiment': 'positive'}
        }
    ]
    
    matching = matcher.get_matching_news(test_news, '半导体', stock_name='中芯国际')
    print(f"匹配到 {len(matching)} 条新闻")
    for m in matching:
        print(f"  - {m.get('title')} (匹配类型: {m.get('match_type')})")
    
    # 测试4: 情感得分计算
    print("\n测试4: 情感得分计算")
    score = matcher.calculate_sentiment_score(matching)
    print(f"情感得分: {score:.2f}")

