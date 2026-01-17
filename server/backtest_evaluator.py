"""
回测评估模块：实现固定持仓期、止盈止损、绩效指标计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class BacktestEvaluator:
    """回测评估器：处理买入/卖出逻辑和绩效计算"""
    
    def __init__(
        self,
        take_profit: float = 0.10,      # 止盈阈值：+10%
        stop_loss: float = -0.05,        # 止损阈值：-5%
        max_hold_days: int = 30,         # 最大持仓天数：30天
        fixed_periods: List[int] = [5, 10, 20, 30]  # 固定持仓期（天）
    ):
        """
        初始化回测评估器
        
        Args:
            take_profit: 止盈阈值（如0.10表示+10%）
            stop_loss: 止损阈值（如-0.05表示-5%）
            max_hold_days: 最大持仓天数
            fixed_periods: 固定持仓期列表（用于多周期评估）
        """
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.fixed_periods = fixed_periods
    
    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        entry_date: str,
        current_date: str,
        current_action: Optional[str] = None
    ) -> Dict:
        """
        判断是否应该卖出
        
        Returns:
            {
                'should_exit': bool,
                'exit_reason': str,
                'exit_price': float,
                'pnl': float,
                'hold_days': int
            }
        """
        # 计算收益率
        pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # 计算持仓天数
        try:
            entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            hold_days = (current_dt - entry_dt).days
        except:
            hold_days = 0
        
        # 1. 止盈检查
        if pnl >= self.take_profit:
            return {
                'should_exit': True,
                'exit_reason': '止盈',
                'exit_price': current_price,
                'pnl': pnl,
                'hold_days': hold_days
            }
        
        # 2. 止损检查
        if pnl <= self.stop_loss:
            return {
                'should_exit': True,
                'exit_reason': '止损',
                'exit_price': current_price,
                'pnl': pnl,
                'hold_days': hold_days
            }
        
        # 3. 时间止损
        if hold_days >= self.max_hold_days:
            return {
                'should_exit': True,
                'exit_reason': '时间止损',
                'exit_price': current_price,
                'pnl': pnl,
                'hold_days': hold_days
            }
        
        # 4. 信号反转（如果提供了当前action）
        if current_action and current_action in ['SELL', 'WAIT']:
            return {
                'should_exit': True,
                'exit_reason': '信号反转',
                'exit_price': current_price,
                'pnl': pnl,
                'hold_days': hold_days
            }
        
        # 继续持仓
        return {
            'should_exit': False,
            'exit_reason': None,
            'exit_price': current_price,
            'pnl': pnl,
            'hold_days': hold_days
        }
    
    def calculate_fixed_period_returns(
        self,
        entry_price: float,
        entry_date: str,
        price_history: pd.DataFrame
    ) -> Dict:
        """
        计算固定持仓期的收益率
        
        Args:
            entry_price: 买入价格
            entry_date: 买入日期
            price_history: 历史价格数据（包含日期和收盘价）
            
        Returns:
            {
                'period_5': {'return': float, 'price': float},
                'period_10': {'return': float, 'price': float},
                ...
            }
        """
        results = {}
        
        try:
            entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
            
            for period in self.fixed_periods:
                target_date = entry_dt + timedelta(days=period)
                target_date_str = target_date.strftime("%Y-%m-%d")
                
                # 在价格历史中查找目标日期
                # 假设price_history有'日期'和'收盘'列
                if '日期' in price_history.columns:
                    # 找到最接近目标日期的价格
                    price_history['日期'] = pd.to_datetime(price_history['日期'])
                    target_dt = pd.to_datetime(target_date_str)
                    
                    # 找到目标日期之后的第一条记录
                    future_prices = price_history[price_history['日期'] >= target_dt]
                    if not future_prices.empty:
                        exit_price = future_prices.iloc[0]['收盘']
                        return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                        results[f'period_{period}'] = {
                            'return': return_pct,
                            'price': exit_price,
                            'date': future_prices.iloc[0]['日期'].strftime("%Y-%m-%d")
                        }
                    else:
                        results[f'period_{period}'] = None
                else:
                    results[f'period_{period}'] = None
        except Exception as e:
            print(f"[BacktestEvaluator] Error calculating fixed period returns: {e}")
        
        return results
    
    def calculate_performance_metrics(
        self,
        performance_data: pd.DataFrame
    ) -> Dict:
        """
        计算绩效指标
        
        Args:
            performance_data: 包含pnl、hold_days等字段的DataFrame
            
        Returns:
            绩效指标字典
        """
        if performance_data.empty or len(performance_data) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'avg_hold_days': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # 基础统计
        total_trades = len(performance_data)
        wins = performance_data[performance_data['pnl'] > 0]
        losses = performance_data[performance_data['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_return = performance_data['pnl'].mean()
        total_return = performance_data['pnl'].sum()
        
        # 持仓时间
        if 'hold_days' in performance_data.columns:
            avg_hold_days = performance_data['hold_days'].mean()
        else:
            avg_hold_days = 0.0
        
        # 夏普比率（简化版：收益率均值/收益率标准差）
        if len(performance_data) > 1:
            returns_std = performance_data['pnl'].std()
            sharpe_ratio = (avg_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤（基于PnL序列）
        cumulative_returns = (1 + performance_data['pnl']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 盈亏比
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # 清理NaN和Inf值，确保JSON可序列化
        def safe_round(value, decimals=2):
            """安全地四舍五入，处理NaN和Inf"""
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return round(float(value), decimals)
            return value
        
        return {
            'total_trades': int(total_trades),
            'win_rate': safe_round(win_rate * 100, 2),  # 转换为百分比
            'avg_return': safe_round(avg_return * 100, 2),  # 转换为百分比
            'total_return': safe_round(total_return * 100, 2),  # 转换为百分比
            'avg_hold_days': safe_round(avg_hold_days, 1),
            'sharpe_ratio': safe_round(sharpe_ratio, 2),
            'max_drawdown': safe_round(max_drawdown * 100, 2),  # 转换为百分比
            'profit_factor': safe_round(profit_factor, 2),
            'avg_win': safe_round(avg_win * 100, 2),  # 转换为百分比
            'avg_loss': safe_round(avg_loss * 100, 2),  # 转换为百分比
            'win_count': int(len(wins)),
            'loss_count': int(len(losses))
        }
    
    def calculate_grouped_metrics(
        self,
        performance_data: pd.DataFrame,
        group_by: str = 'score_range'
    ) -> Dict:
        """
        按分组计算绩效指标
        
        Args:
            performance_data: 绩效数据
            group_by: 分组字段（score_range, industry, exit_reason等）
            
        Returns:
            分组绩效指标
        """
        if performance_data.empty:
            return {}
        
        grouped_metrics = {}
        
        if group_by == 'score_range':
            # 按得分区间分组
            def get_score_range(score):
                if score >= 90:
                    return '90-100'
                elif score >= 85:
                    return '85-90'
                elif score >= 80:
                    return '80-85'
                else:
                    return '<80'
            
            performance_data['score_range'] = performance_data['score'].apply(get_score_range)
            group_key = 'score_range'
        else:
            group_key = group_by
        
        if group_key not in performance_data.columns:
            return {}
        
        for group_name, group_data in performance_data.groupby(group_key):
            metrics = self.calculate_performance_metrics(group_data)
            grouped_metrics[group_name] = metrics
        
        return grouped_metrics
    
    def calculate_relative_return(
        self,
        stock_return: float,
        market_return: float
    ) -> float:
        """
        计算相对市场收益
        
        Args:
            stock_return: 股票收益率
            market_return: 市场收益率（如沪深300）
            
        Returns:
            相对收益 = 股票收益 - 市场收益
        """
        return stock_return - market_return

if __name__ == "__main__":
    # 测试代码
    evaluator = BacktestEvaluator()
    
    # 测试卖出判断
    result = evaluator.should_exit(
        entry_price=10.0,
        current_price=11.5,  # +15%
        entry_date="2024-01-01",
        current_date="2024-01-05"
    )
    print("卖出判断测试:", result)
    
    # 测试绩效指标
    test_data = pd.DataFrame({
        'pnl': [0.1, -0.05, 0.15, 0.08, -0.02],
        'hold_days': [5, 3, 10, 7, 2],
        'score': [90, 85, 95, 88, 82]
    })
    metrics = evaluator.calculate_performance_metrics(test_data)
    print("\n绩效指标测试:", metrics)

