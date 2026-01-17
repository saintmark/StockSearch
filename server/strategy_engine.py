import pandas as pd
import numpy as np
import math
from typing import List, Dict, Optional

class StrategyEngine:
    """核心策略引擎：多因子选股模型（技术+基本面+风险+消息面）"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标 (SMA, MACD, RSI, KDJ, 成交量指标)"""
        if df.empty or len(df) < 26:
            return df
            
        # 1. 移动平均线 (SMA)
        df['ma20'] = df['收盘'].rolling(window=20).mean()
        df['ma60'] = df['收盘'].rolling(window=60).mean()
        df['ma120'] = df['收盘'].rolling(window=120).mean()
        
        # 2. MACD
        exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['dif'] = exp1 - exp2
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd'] = (df['dif'] - df['dea']) * 2
        
        # 3. RSI (相对强弱指标)
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. KDJ (随机指标)
        low_min = df['最低'].rolling(window=9).min()
        high_max = df['最高'].rolling(window=9).max()
        rsv = (df['收盘'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 5. 成交量指标
        if '成交量' in df.columns:
            # OBV (能量潮)
            df['obv'] = (np.sign(df['收盘'].diff()) * df['成交量']).fillna(0).cumsum()
            
            # 量比 (当日成交量 / 过去5日平均成交量)
            df['volume_ma5'] = df['成交量'].rolling(window=5).mean()
            df['volume_ratio'] = df['成交量'] / df['volume_ma5']
            
            # 成交量均线
            df['volume_ma20'] = df['成交量'].rolling(window=20).mean()
        else:
            df['obv'] = 0
            df['volume_ratio'] = 1.0
            df['volume_ma20'] = 0
        
        # 6. 布林带
        df['bb_middle'] = df['收盘'].rolling(window=20).mean()
        bb_std = df['收盘'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
        """计算风险指标：波动率、最大回撤"""
        if df.empty or len(df) < 20:
            return {
                'volatility_20': 0.0,
                'volatility_60': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0
            }
        
        # 1. 历史波动率 (年化)
        returns = df['收盘'].pct_change()
        volatility_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
        volatility_60 = returns.rolling(window=60).std().iloc[-1] * np.sqrt(252) * 100
        
        # 2. 最大回撤
        cumulative_max = df['收盘'].cummax()
        drawdown = (df['收盘'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100  # 转换为百分比
        current_drawdown = drawdown.iloc[-1] * 100
        
        return {
            'volatility_20': float(volatility_20) if not np.isnan(volatility_20) else 0.0,
            'volatility_60': float(volatility_60) if not np.isnan(volatility_60) else 0.0,
            'max_drawdown': float(max_drawdown) if not np.isnan(max_drawdown) else 0.0,
            'current_drawdown': float(current_drawdown) if not np.isnan(current_drawdown) else 0.0
        }
    
    @staticmethod
    def calculate_fundamental_score(finance_data: Dict) -> float:
        """计算基本面得分 (0-100)"""
        if not finance_data:
            return 50.0  # 默认中性分
        
        score = 50.0  # 基础分
        reasons = []
        
        # 1. 价值因子 (PE, PB)
        try:
            pe = finance_data.get('市盈率', '--')
            pb = finance_data.get('市净率', '--')
            
            if pe != '--' and isinstance(pe, (int, float)) and pe > 0:
                # PE越低越好，但也要考虑合理性
                if 0 < pe < 15:
                    score += 10  # 低估
                    reasons.append(f"PE较低({pe:.2f})")
                elif 15 <= pe < 30:
                    score += 5  # 合理
                elif pe > 50:
                    score -= 5  # 高估
            
            if pb != '--' and isinstance(pb, (int, float)) and pb > 0:
                if 0 < pb < 2:
                    score += 10  # 低估
                    reasons.append(f"PB较低({pb:.2f})")
                elif 2 <= pb < 5:
                    score += 5  # 合理
                elif pb > 8:
                    score -= 5  # 高估
        except:
            pass
        
        # 2. 质量因子 (ROE, 毛利率)
        try:
            roe = finance_data.get('净资产收益率', '--')
            if roe != '--' and isinstance(roe, (int, float)):
                if roe > 15:
                    score += 15  # 高质量
                    reasons.append(f"ROE较高({roe:.2f}%)")
                elif roe > 10:
                    score += 8
                elif roe < 5:
                    score -= 10  # 低质量
            
            gross_margin = finance_data.get('销售毛利率', '--')
            if gross_margin != '--' and isinstance(gross_margin, (int, float)):
                if gross_margin > 30:
                    score += 10
                    reasons.append(f"毛利率较高({gross_margin:.2f}%)")
                elif gross_margin < 15:
                    score -= 5
        except:
            pass
        
        return max(0, min(100, score))

    def generate_recommendation(
        self, 
        kline_df: pd.DataFrame, 
        sentiment_score: float = 0.0,
        tech_weight: float = 0.6,
        sentiment_weight: float = 0.2,
        fundamental_weight: float = 0.15,
        risk_weight: float = 0.05,
        finance_data: Optional[Dict] = None,
        weekly_kline_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        生成投资建议
        kline_df: 包含历史 K 线的 DataFrame
        sentiment_score: 归一化后的舆情得分 (-1 到 1)
        tech_weight/sentiment_weight: 由自迭代模块提供的动态权重
        """
        if kline_df.empty or len(kline_df) < 5:
            return {
                "symbol": "Unknown",
                "action": "WAIT", 
                "reason": "数据不足",
                "advice": "等待数据同步",
                "score": 50,
                "price": 0.0,
                "reasons": ["数据不足"]
            }

        # 计算指标
        df = self.calculate_indicators(kline_df.copy())
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        current_price = last_row['收盘']
        reasons = []
        
        # ========== 1. 技术因子评分 ==========
        
        # 1.1 趋势因子 (均线系统 + 多时间框架确认)
        ma20 = last_row.get('ma20', 0)
        ma60 = last_row.get('ma60', 0)
        ma120 = last_row.get('ma120', 0)
        
        # 日线趋势
        daily_trend_up = current_price > ma20 > ma60
        daily_trend_strong = current_price > ma60 > ma120
        
        # 周线趋势确认（如果提供了周线数据）
        weekly_trend_up = True
        if weekly_kline_df is not None and not weekly_kline_df.empty:
            weekly_df = self.calculate_indicators(weekly_kline_df.copy())
            if len(weekly_df) > 0:
                weekly_last = weekly_df.iloc[-1]
                weekly_ma5 = weekly_last.get('ma20', 0)  # 周线MA20相当于5周均线
                weekly_ma20 = weekly_last.get('ma60', 0)  # 周线MA60相当于20周均线
                weekly_trend_up = weekly_last['收盘'] > weekly_ma5 > weekly_ma20
        
        # 多周期共振加分
        trend_score = 50.0
        if daily_trend_strong and weekly_trend_up:
            trend_score = 85.0  # 多周期共振，强势
            reasons.append("多周期共振：日线+周线均呈上升趋势")
        elif daily_trend_up:
            trend_score = 70.0
            reasons.append("日线趋势向上")
        elif current_price > ma60:
            trend_score = 60.0
            reasons.append("价格位于60日均线上方")
        else:
            trend_score = 40.0
            reasons.append("趋势偏弱")
        
        # 1.2 动量因子 (MACD + RSI + KDJ)
        macd_val = last_row.get('macd', 0)
        prev_macd = prev_row.get('macd', 0)
        macd_slope = macd_val - prev_macd
        
        rsi = last_row.get('rsi', 50)
        k = last_row.get('k', 50)
        d = last_row.get('d', 50)
        
        momentum_score = 50.0
        if last_row['dif'] > last_row['dea'] and macd_slope > 0:
            momentum_score = 80.0
            reasons.append("MACD多头加速")
        elif last_row['dif'] > last_row['dea']:
            momentum_score = 65.0
        else:
            momentum_score = 40.0
        
        # RSI和KDJ确认
        if 30 < rsi < 70:  # RSI在合理区间
            momentum_score += 5
        if k > d and k < 80:  # KDJ金叉且未超买
            momentum_score += 5
        elif k > 80:
            momentum_score -= 5  # 超买
        elif k < 20:
            momentum_score += 5  # 超卖反弹
        
        momentum_score = max(20, min(95, momentum_score))
        reasons.append(f"RSI: {rsi:.1f}, KDJ: K={k:.1f} D={d:.1f}")
        
        # 1.3 成交量因子
        volume_score = 50.0
        if 'volume_ratio' in last_row:
            volume_ratio = last_row.get('volume_ratio', 1.0)
            price_change = (current_price - prev_row['收盘']) / prev_row['收盘'] * 100
            
            if volume_ratio > 1.5 and price_change > 0:
                volume_score = 85.0  # 放量上涨
                reasons.append(f"放量上涨(量比{volume_ratio:.2f})")
            elif volume_ratio > 1.2 and price_change > 0:
                volume_score = 70.0
                reasons.append(f"温和放量(量比{volume_ratio:.2f})")
            elif volume_ratio < 0.8 and price_change < 0:
                volume_score = 60.0  # 缩量下跌，可能洗盘
                reasons.append(f"缩量下跌(量比{volume_ratio:.2f})")
            elif volume_ratio < 0.5:
                volume_score = 40.0  # 极度缩量，流动性差
                reasons.append(f"极度缩量(量比{volume_ratio:.2f})")
        
        # 技术因子综合得分
        tech_score = (trend_score * 0.4) + (momentum_score * 0.4) + (volume_score * 0.2)
        
        # ========== 2. 风险因子评分 ==========
        risk_metrics = self.calculate_risk_metrics(df)
        volatility_20 = risk_metrics['volatility_20']
        max_drawdown = risk_metrics['max_drawdown']
        current_drawdown = risk_metrics['current_drawdown']
        
        # 风险调整：波动率越高，风险得分越低
        risk_score = 100.0
        if volatility_20 > 50:
            risk_score = 30.0  # 极高波动
            reasons.append(f"风险：波动率极高({volatility_20:.1f}%)")
        elif volatility_20 > 35:
            risk_score = 50.0
            reasons.append(f"风险：波动率较高({volatility_20:.1f}%)")
        elif volatility_20 < 15:
            risk_score = 90.0  # 低波动，风险低
            reasons.append(f"风险：波动率较低({volatility_20:.1f}%)")
        
        # 回撤惩罚
        if current_drawdown < -20:
            risk_score -= 20  # 当前回撤超过20%
            reasons.append(f"风险：当前回撤{current_drawdown:.1f}%")
        if max_drawdown < -40:
            risk_score -= 15  # 历史最大回撤超过40%
            reasons.append(f"风险：历史最大回撤{max_drawdown:.1f}%")
        
        risk_score = max(0, min(100, risk_score))
        
        # ========== 3. 基本面因子评分 ==========
        fundamental_score = 50.0
        if finance_data:
            fundamental_score = self.calculate_fundamental_score(finance_data)
            if fundamental_score > 70:
                reasons.append("基本面：优质公司")
            elif fundamental_score < 40:
                reasons.append("基本面：需关注")
        else:
            reasons.append("基本面：数据缺失")
        
        # ========== 4. 消息面因子 ==========
        sent_impact = (sentiment_score + 1) * 50  # 映射到 0-100
        
        # ========== 5. 综合评分 ==========
        # 多因子加权平均
        final_score = (
            tech_score * tech_weight +
            sent_impact * sentiment_weight +
            fundamental_score * fundamental_weight +
            risk_score * risk_weight
        )
        
        # 风险调整：如果风险过高，降低最终得分
        if volatility_20 > 40 or current_drawdown < -25:
            final_score *= 0.8  # 高风险惩罚
            reasons.append("风险调整：高风险股票已降分")
        
        # ========== 6. 防御性检查 ==========
        if math.isnan(final_score) or math.isinf(final_score):
            final_score = 50.0
            reasons.append("提示：量化指标计算受限 (数据异常)")
        
        # 增加微小波动以辅助前端排序
        symbol_hash = (sum(ord(c) for c in str(last_row.get('代码', ''))) % 100) / 1000.0
        final_score += symbol_hash
        
        final_score = max(0, min(100, final_score))
        
        # 添加风险指标到返回结果
        risk_info = {
            'volatility_20': round(volatility_20, 2),
            'max_drawdown': round(max_drawdown, 2),
            'current_drawdown': round(current_drawdown, 2)
        }
        
        # 5. 生成建议文字
        if final_score >= 85:
            action = "BUY"
            advice = "核心指标显著爆发，多头动能强劲，建议顺势入场。"
        elif final_score >= 60:
            action = "HOLD"
            advice = "表现尚可但处于整理阶段，建议持仓并设置保护性止损。"
        elif final_score <= 40:
            action = "SELL"
            advice = "动能指标显著恶化，建议及时减仓锁定利润或止损。"
        else:
            action = "WAIT"
            advice = "目前处于震荡格局，信号不明确，优先保持现金仓位。"
            
        return {
            "symbol": last_row.get('代码', '未知'),
            "score": round(final_score, 1),
            "action": action,
            "advice": advice,
            "reasons": reasons,
            "price": current_price,
            "risk_metrics": risk_info,
            "factor_scores": {
                "tech": round(tech_score, 1),
                "fundamental": round(fundamental_score, 1),
                "sentiment": round(sent_impact, 1),
                "risk": round(risk_score, 1)
            }
        }

if __name__ == "__main__":
    # 模拟测试
    engine = StrategyEngine()
    data = {
        '收盘': [10, 11, 12, 11.5, 13, 14, 15, 14.5, 16] * 20,
        '最低': [9]*180, '最高': [17]*180, '开盘': [10]*180
    }
    df = pd.DataFrame(data)
    print(engine.generate_recommendation(df, 0.5))
