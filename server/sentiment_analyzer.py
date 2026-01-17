import re
from typing import List, Dict

class SentimentAnalyzer:
    """升级版 A 股语义诊断引擎：融合 LLM 深度分析与规则兜底机制"""
    
    def __init__(self):
        # 0. 初始化 LLM 客户端
        try:
            from llm_client import LLMClient
            self.llm_client = LLMClient()
        except ImportError:
            self.llm_client = None
            print("[SentimentAnalyzer] Warning: LLMClient not found. Running in legacy mode.")

        # 1. 行业 DNA 关键词矩阵 (Legacy Fallback)
        self.sector_keywords = {
            '半导体/芯片': ['半导体', '芯片', '光刻机', '中芯', '先进制程', '封测'],
            '人工智能': ['AI', '大模型', '算力', '英伟达', '生成式', 'CPO', 'ChatGPT'],
            '新能源': ['锂电', '固态电池', '宁德', '光伏', '储能', '低空经济'],
            '金融/地产': ['银行', '地产', '重组', '降息', '降准', '平安', '万科'],
            '大消费': ['白酒', '餐饮', '社零', '茅台', '消费电子'],
            '宏观经济': ['GDP', 'CPI', '财政', '美联储', '加息', '关税']
        }
        
        # 2. 正向/负向词库 (Legacy Fallback)
        self.pos_words = {'增长', '突破', '提振', '爆发', '新高', '买入', '增持', '中标', '向好', '盈余'}
        self.neg_words = {'下滑', '暴跌', '卖出', '减持', '降级', '亏损', '告负', '回落', '质疑', '立案'}
        self.neg_context = {'暴乱', '冲突', '危机', '崩溃', '罢工', '事故', '风险', '疫情'}

    def analyze_text(self, text: str) -> Dict:
        """多维深度分析：优先尝试 LLM，失败则回退到规则引擎"""
        if not text:
            return {"score": 0, "sentiment": "neutral", "sector": "全市场", "reasoning": "无内容"}
            
        # Strategy A: LLM Deep Analysis
        if self.llm_client:
            llm_result = self.llm_client.analyze_news(text)
            if llm_result:
                return llm_result
        
        # Strategy B: Rule-based Fallback (Legacy Logic)
        return self._analyze_legacy(text)

    def _analyze_legacy(self, text: str) -> Dict:
        """基于关键词的传统分析逻辑 (兜底用)"""
        # A. 行业侦测
        target_sector = "全市场"
        for sector, keys in self.sector_keywords.items():
            if any(k in text for k in keys):
                target_sector = sector
                break
        
        # B. 语境探测
        has_negative_context = any(word in text for word in self.neg_context)
        neg_context_words = [word for word in self.neg_context if word in text]
        
        # C. 情感计分
        found_pos = [w for w in self.pos_words if w in text]
        found_neg = [w for w in self.neg_words if w in text]
        
        pos_weight = 1.0
        neg_weight = 1.0
        
        if has_negative_context:
            pos_weight = -0.5 
            reason_msg = f"[规则兜底] 检测到负面语境({', '.join(neg_context_words)})，已反转正向词影响。"
        else:
            reason_msg = "[规则兜底] 基于行业关键词与语义极性打分。"

        score_raw = (len(found_pos) * pos_weight) - (len(found_neg) * neg_weight)
        
        total_hits = len(found_pos) + len(found_neg) + len(neg_context_words)
        if total_hits == 0:
            score = 0
        else:
            score = max(-1, min(1, score_raw / (total_hits if total_hits > 0 else 1)))

        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "score": round(score, 2),
            "sentiment": sentiment,
            "sector": target_sector,
            "pos_words": found_pos, # Keep these for legacy debugging if needed
            "neg_words": found_neg + neg_context_words,
            "reasoning": reason_msg
        }

    def batch_analyze(self, news_list: List[Dict]) -> List[Dict]:
        """批量处理接口"""
        import concurrent.futures
        
        # 使用线程池并发调用 LLM 以减少总耗时
        # 注意：如果 LLM Client 内部没有处理好并发限流，这里可能会报错，
        # 但我们在 LLMClient 里加了 retry 和 simple backoff
        results = [None] * len(news_list)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_index = {}
            for i, item in enumerate(news_list):
                content = item.get('内容') or item.get('content') or ""
                future = executor.submit(self.analyze_text, content)
                future_to_index[future] = i
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    res = future.result()
                    results[index] = res
                except Exception as e:
                    print(f"Batch analyze error at index {index}: {e}")
                    results[index] = {
                        "score": 0, "sentiment": "neutral", 
                        "sector": "未知", "reasoning": "分析失败"
                    }

        # Merge results back to items
        for i, item in enumerate(news_list):
            if results[i]:
                # Flatten the result into the item or keep it nested?
                # The original code did: item['sentiment'] = self.analyze_text(content)
                # But looking at main.py: news_pool = analyzer.batch_analyze(...)
                # And get_news_flash returns: processed_news
                # Let's check main.py usage again.
                # In main.py: 
                # news_pool = analyzer.batch_analyze(processed_news_for_analyzer)
                # ... mentions = [n for n in news_pool if ... stock_name in n['content']]
                # Wait, batch_analyze in original code returned news_list with 'sentiment' field added?
                # Original: item['sentiment'] = self.analyze_text(content); return news_list
                # So I should do the same.
                item.update(results[i]) # Flatten it? or item['sentiment'] = results[i]?
                # The LLM result has 'score', 'sentiment', 'sector', 'reasoning'.
                # The original result of analyze_text was a dict.
                # The original batch_analyze did: item['sentiment'] = dict
                # So access was item['sentiment']['score'] etc.
                # Let's stick to original structure to avoid breaking main.py?
                # main.py usage:
                # mentions = [n for n in news_pool if ... stock_name in n['content']]
                # dynamic_sent = min(0.4, 0.2 * len(mentions))
                # It doesn't seem to use the inner score of news items for stock scanning!
                # It only counts Mentions!
                # But get_news_flash returns processed_news to frontend.
                # Frontend probably displays it.
                # If I flatten it, frontend might like it better, but risk breaking compatibility if frontend expects {sentiment: {...}}
                # Let's keep it nested as 'sentiment' field to be safe, OR check frontend code.
                # But wait, the user wants "Reasoning" displayed.
                # I'll put the analysis result in 'analysis' field or merge it.
                # Let's follow the original pattern: item['sentiment'] = result
                item['sentiment'] = results[i]
                
        return news_list

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    test_text = "多部委发文支持高新区发展，新质生产力板块异动。某公司获得重大中标订单，业绩增长可期。"
    print(analyzer.analyze_text(test_text))
