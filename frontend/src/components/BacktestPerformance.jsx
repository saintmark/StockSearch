import React from 'react';
import { TrendingUp, TrendingDown, Clock, Target, AlertCircle } from 'lucide-react';

const BacktestPerformance = ({ performance }) => {
  if (!performance) {
    return (
      <div className="text-center py-20 text-gray-600 italic font-medium">
        当前无回测数据，等待算法产生推荐并完成交易...
      </div>
    );
  }

  const { performance_data = [], overall_metrics = {}, grouped_metrics = {} } = performance;

  return (
    <div className="space-y-8">
      {/* 整体绩效指标卡片 */}
      {Object.keys(overall_metrics).length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-2">
                <Target className="text-primary" size={20} />
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">总交易次数</div>
              </div>
              <div className="text-3xl font-black text-white">{overall_metrics.total_trades || 0}</div>
            </div>
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-2">
                {overall_metrics.win_rate >= 50 ? (
                  <TrendingUp className="text-success" size={20} />
                ) : (
                  <TrendingDown className="text-danger" size={20} />
                )}
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">胜率</div>
              </div>
              <div className={`text-3xl font-black ${overall_metrics.win_rate >= 50 ? 'text-success' : 'text-danger'}`}>
                {overall_metrics.win_rate || 0}%
              </div>
              <div className="text-xs text-gray-500 mt-2">
                盈利: <span className="text-success font-bold">{overall_metrics.win_count || 0}</span> | 
                亏损: <span className="text-danger font-bold">{overall_metrics.loss_count || 0}</span>
              </div>
            </div>
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-2">
                {overall_metrics.avg_return >= 0 ? (
                  <TrendingUp className="text-success" size={20} />
                ) : (
                  <TrendingDown className="text-danger" size={20} />
                )}
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">平均收益率</div>
              </div>
              <div className={`text-3xl font-black ${overall_metrics.avg_return >= 0 ? 'text-success' : 'text-danger'}`}>
                {overall_metrics.avg_return || 0}%
              </div>
              <div className="text-xs text-gray-500 mt-2">
                总收益: <span className={overall_metrics.total_return >= 0 ? 'text-success' : 'text-danger'}>
                  {overall_metrics.total_return || 0}%
                </span>
              </div>
            </div>
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-2">
                <Clock className="text-warning" size={20} />
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">平均持仓天数</div>
              </div>
              <div className="text-3xl font-black text-white">{overall_metrics.avg_hold_days || 0}</div>
              <div className="text-xs text-gray-500 mt-2">
                夏普比率: <span className="text-white font-bold">{overall_metrics.sharpe_ratio || 0}</span>
              </div>
            </div>
          </div>

          {/* 风险指标 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-2">
                <AlertCircle className="text-danger" size={20} />
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">最大回撤</div>
              </div>
              <div className={`text-2xl font-black ${overall_metrics.max_drawdown < -10 ? 'text-danger' : 'text-warning'}`}>
                {overall_metrics.max_drawdown || 0}%
              </div>
            </div>
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="text-xs font-black text-gray-400 uppercase tracking-widest mb-2">盈亏比</div>
              <div className={`text-2xl font-black ${overall_metrics.profit_factor >= 1.5 ? 'text-success' : 'text-warning'}`}>
                {overall_metrics.profit_factor || 0}
              </div>
              <div className="text-xs text-gray-500 mt-2 space-y-1">
                <div>平均盈利: <span className="text-success font-bold">{overall_metrics.avg_win || 0}%</span></div>
                <div>平均亏损: <span className="text-danger font-bold">{overall_metrics.avg_loss || 0}%</span></div>
              </div>
            </div>
            <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
              <div className="text-xs font-black text-gray-400 uppercase tracking-widest mb-2">风险指标</div>
              <div className="text-sm text-gray-400 space-y-2">
                <div className="flex justify-between">
                  <span>夏普比率:</span>
                  <span className="text-white font-bold">{overall_metrics.sharpe_ratio || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>最大回撤:</span>
                  <span className="text-white font-bold">{overall_metrics.max_drawdown || 0}%</span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* 分组统计 */}
      {Object.keys(grouped_metrics).length > 0 && (
        <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl">
          <h3 className="text-xl font-black text-white mb-4">按得分区间统计</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(grouped_metrics).map(([range, metrics]) => (
              <div key={range} className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
                <div className="text-xs font-black text-primary mb-2">{range}分</div>
                <div className="text-sm space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">胜率:</span>
                    <span className={`font-bold ${metrics.win_rate >= 50 ? 'text-success' : 'text-danger'}`}>
                      {metrics.win_rate || 0}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">平均收益:</span>
                    <span className={`font-bold ${metrics.avg_return >= 0 ? 'text-success' : 'text-danger'}`}>
                      {metrics.avg_return || 0}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">交易次数:</span>
                    <span className="font-bold text-white">{metrics.total_trades || 0}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 详细交易记录 */}
      <div className="bg-card border border-gray-800 rounded-2xl p-6 shadow-xl overflow-hidden">
        <h3 className="text-xl font-black text-white mb-6">交易明细</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="px-6 py-4 text-left text-xs font-black text-gray-400 uppercase tracking-widest">股票代码</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">买入日期</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">买入价</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">卖出日期</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">卖出价</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">持仓天数</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">收益率</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">卖出原因</th>
                <th className="px-6 py-4 text-center text-xs font-black text-gray-400 uppercase tracking-widest">状态</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {performance_data.length > 0 ? (
                performance_data.map((item, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
                    <td className="px-6 py-4">
                      <div className="font-black text-primary text-lg">{item.symbol}</div>
                      {item.industry && (
                        <div className="text-[10px] text-gray-500 mt-1">{item.industry}</div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-gray-300">
                      {item.entry_date || item.created_at ? (item.entry_date || item.created_at).substring(0, 10) : '--'}
                    </td>
                    <td className="px-6 py-4 text-center font-bold text-white">
                      {item.entry_price || item.price || '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-gray-300">
                      {item.exit_date ? item.exit_date.substring(0, 10) : '--'}
                    </td>
                    <td className="px-6 py-4 text-center font-bold text-gray-300">
                      {item.exit_price || item.close_price || '--'}
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-gray-300">
                      {item.hold_days ? `${item.hold_days}天` : '--'}
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className={`text-lg font-black ${(item.pnl || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {item.pnl !== null && item.pnl !== undefined ? (item.pnl * 100).toFixed(2) + '%' : '--'}
                      </span>
                      {item.max_profit !== null && item.max_profit !== undefined && (
                        <div className="text-[10px] text-success mt-1">最大盈利: {(item.max_profit * 100).toFixed(2)}%</div>
                      )}
                      {item.max_loss !== null && item.max_loss !== undefined && (
                        <div className="text-[10px] text-danger mt-1">最大亏损: {(item.max_loss * 100).toFixed(2)}%</div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-center">
                      {item.exit_reason ? (
                        <span className={`px-2 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${
                          item.exit_reason === '止盈' ? 'bg-success/20 text-success border border-success/30' :
                          item.exit_reason === '止损' ? 'bg-danger/20 text-danger border border-danger/30' :
                          item.exit_reason === '时间止损' ? 'bg-warning/20 text-warning border border-warning/30' :
                          'bg-gray-800 text-gray-400 border border-gray-700'
                        }`}>
                          {item.exit_reason}
                        </span>
                      ) : (
                        <span className="text-gray-500 text-xs">--</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest border ${
                        item.status === 'OPEN' ? 'bg-warning/10 text-warning border-warning/20' : 
                        'bg-gray-800 text-gray-500 border-gray-700'
                      }`}>
                        {item.status || 'CLOSED'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="9" className="px-8 py-20 text-center text-gray-600 italic font-medium">
                    当前无回测数据，等待算法产生推荐并完成交易...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BacktestPerformance;



