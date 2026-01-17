import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { LayoutDashboard, Newspaper, TrendingUp, History, Settings, Search, Bell, RefreshCw, BarChart3, AlertCircle, X, ExternalLink, Info, XCircle, ArrowUpCircle, ArrowDownCircle, Filter, ArrowUp, ArrowDown, ChevronRight, PieChart } from 'lucide-react';
import { initStockChart } from './utils/chartUtils';
import IndustryPieChart from './components/IndustryPieChart';
import BacktestPerformance from './components/BacktestPerformance';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [news, setNews] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [performance, setPerformance] = useState(null); // Changed from [] to null based on user's snippet
  const [loadingNews, setLoadingNews] = useState(false);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [searchSymbol, setSearchSymbol] = useState('');
  const [klineData, setKlineData] = useState([]);
  const [weights, setWeights] = useState({ tech_weight: 0.8, sentiment_weight: 0.2 });
  const [savingWeights, setSavingWeights] = useState(false);
  const [selectedRec, setSelectedRec] = useState(null);
  const [financeData, setFinanceData] = useState(null);
  const [individualInfo, setIndividualInfo] = useState(null);
  const [modalData, setModalData] = useState([]);
  const [loadingChart, setLoadingChart] = useState(false);
  const [scoreDetail, setScoreDetail] = useState(null);
  const [fullRank, setFullRank] = useState([]); // 新增：全市场排行榜数据
  const [loadingRank, setLoadingRank] = useState(false);
  const [rankPagination, setRankPagination] = useState({ total: 0, page: 1, page_size: 500, total_pages: 0 });
  const [industryStats, setIndustryStats] = useState([]); // 新增：行业统计数据
  const [sectors, setSectors] = useState([]); // 新增：全行业列表

  // 新增：排行榜筛选与排序状态（服务端分页需要状态提升）
  const [rankFilter, setRankFilter] = useState({ sector: '', minPrice: '', maxPrice: '', search: '' });
  const [rankSort, setRankSort] = useState({ key: 'score', direction: 'desc' });

  const chartRef = useRef(null);
  const modalChartRef = useRef(null);

  // 数据获取函数
  const fetchData = async () => {
    setLoadingNews(true);
    setLoadingRecs(true);
    try {
      const newsRes = await axios.get(`${API_BASE}/news/flash`);
      const sortedNews = Array.isArray(newsRes.data) ? [...newsRes.data].reverse() : [];
      setNews(sortedNews);

      const recRes = await axios.get(`${API_BASE}/stocks/market_recommendations`);
      setRecommendations(Array.isArray(recRes.data) ? recRes.data : []);

      const perfRes = await axios.get(`${API_BASE}/review/performance`);
      // 新API返回格式：{ performance_data, overall_metrics, grouped_metrics }
      if (perfRes.data && typeof perfRes.data === 'object' && !Array.isArray(perfRes.data)) {
        setPerformance(perfRes.data);
      } else {
        // 兼容旧格式
        setPerformance({
          performance_data: Array.isArray(perfRes.data) ? perfRes.data : [],
          overall_metrics: {},
          grouped_metrics: {}
        });
      }

      const weightRes = await axios.get(`${API_BASE}/review/weights`);
      if (weightRes.data) setWeights(weightRes.data);

      // 获取排行榜数据（第1页）
      fetchRankData(1);
    } catch (err) {
      console.error("Data fetch error", err);
    } finally {
      setLoadingNews(false);
      setLoadingRecs(false);
    }
  };

  // 获取排行榜分页数据 (支持服务端过滤)
  const fetchRankData = async (page = 1, currentFilter = rankFilter, currentSort = rankSort) => {
    setLoadingRank(true);
    try {
      const params = {
        page,
        page_size: 500,
        search: currentFilter.search || undefined,
        industry: currentFilter.sector || undefined,
        min_price: currentFilter.minPrice || undefined,
        max_price: currentFilter.maxPrice || undefined,
        sort_by: currentSort.key,
        sort_dir: currentSort.direction
      };

      const rankRes = await axios.get(`${API_BASE}/stocks/full_rank`, { params });

      if (rankRes.data) {
        setFullRank(Array.isArray(rankRes.data.data) ? rankRes.data.data : []);
        setRankPagination({
          total: rankRes.data.total || 0,
          page: rankRes.data.page || 1,
          page_size: rankRes.data.page_size || 500,
          total_pages: rankRes.data.total_pages || 0
        });
      }
    } catch (err) {
      console.error("Rank data fetch error", err);
    } finally {
      setLoadingRank(false);
    }
  };

  // 监听筛选条件变化（防抖 500ms）
  useEffect(() => {
    const handler = setTimeout(() => {
      // 当筛选条件变化时，重置回第 1 页
      fetchRankData(1, rankFilter, rankSort);
    }, 500);
    return () => clearTimeout(handler);
  }, [rankFilter, rankSort]); // 排序变化也触发，但不需要重置页码？通常需要重置

  // 页码切换处理
  const handlePageChange = (newPage) => {
    fetchRankData(newPage, rankFilter, rankSort);
  };

  // 新增：获取行业分布统计
  const fetchIndustryStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stocks/industry_distribution?limit=200`);
      if (res.data) {
        setIndustryStats(res.data);
      }
    } catch (err) {
      console.error("Industry stats fetch error", err);
    }
  };

  // 监听 Tab 切换，加载行业数据
  useEffect(() => {
    if (activeTab === 'rank') {
      fetchIndustryStats();
      fetchIndustries();
    }
  }, [activeTab]);

  const fetchIndustries = async () => {
    try {
      const res = await axios.get(`${API_BASE}/stocks/industries`);
      if (Array.isArray(res.data)) {
        setSectors(res.data);
      }
    } catch (err) {
      console.error("Fetch industries error", err);
    }
  };

  const fetchKLine = async (symbol = "000001") => {
    try {
      const res = await axios.get(`${API_BASE}/stocks/kline/${symbol}?days=200`);
      setKlineData(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.error("KLine error", err);
    }
  };

  useEffect(() => {
    let timer;
    const autoFetch = async () => {
      await fetchData();
      // 下一次请求在当前请求完成后 60 秒触发
      timer = setTimeout(autoFetch, 60000);
    };

    fetchKLine(); // Initial call for kline data
    autoFetch(); // Start the recurring data fetch

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (klineData.length > 0 && chartRef.current && activeTab === 'dashboard') {
      initStockChart(chartRef.current, klineData);
    }
  }, [klineData, activeTab]);

  useEffect(() => {
    if (selectedRec && modalData.length > 0 && modalChartRef.current) {
      setTimeout(() => {
        initStockChart(modalChartRef.current, modalData);
      }, 300); // 确保 Modal 动画完成
    }
  }, [selectedRec, modalData]);

  const marketSentiment = React.useMemo(() => {
    if (!news.length) return { score: 0, trend: '0.00' };
    const scores = news.filter(n => n.sentiment).map(n => n.sentiment.score);
    if (!scores.length) return { score: 0, trend: '0.00' };
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    return { score: avg.toFixed(2), trend: (avg * 10).toFixed(2) + "%" };
  }, [news]);

  const handleSearch = async (e) => {
    if (e.key === 'Enter' && searchSymbol) {
      fetchKLine(searchSymbol);
      axios.get(`${API_BASE}/stocks/recommend/${searchSymbol}`).catch(() => { });
    }
  };

  const handleSaveWeights = async () => {
    setSavingWeights(true);
    try {
      await axios.post(`${API_BASE}/review/update_weights`, weights);
      alert("参数保存成功，策略已实时更新！");
    } catch (err) {
      alert("保存失败: " + err.message);
    } finally {
      setSavingWeights(false);
    }
  };

  const handleIterate = async () => {
    try {
      const res = await axios.get(`${API_BASE}/review/iterate`);
      alert(res.data.msg || (res.data.status === 'optimized' ? "进化成功" : "样本不足"));
    } catch (err) {
      alert("迭代尝试失败");
    }
  };

  const handleWeightChange = (type, val) => {
    const value = parseFloat(val);
    if (type === 'tech') {
      setWeights({ tech_weight: value, sentiment_weight: 1 - value });
    } else {
      setWeights({ tech_weight: 1 - value, sentiment_weight: value });
    }
  };

  const handleSelectRec = async (rec) => {
    setSelectedRec(rec);
    setFinanceData(null);
    setIndividualInfo(null);
    setModalData([]);
    setLoadingChart(true);

    // 1. 获取核心数据 (K线 + 基础信息)
    try {
      const [klineRes, infoRes] = await Promise.all([
        axios.get(`${API_BASE}/stocks/kline/${rec.symbol}?days=250`),
        axios.get(`${API_BASE}/stocks/info/${rec.symbol}`)
      ]);

      setModalData(Array.isArray(klineRes.data) ? klineRes.data : []);
      setIndividualInfo(infoRes.data);
    } catch (err) {
      console.error("Fetch primary data error", err);
    } finally {
      setLoadingChart(false);
    }

    // 2. 获取财务数据 (次要)
    try {
      const financeRes = await axios.get(`${API_BASE}/stocks/finance/${rec.symbol}`);
      if (Array.isArray(financeRes.data) && financeRes.data.length > 0) {
        setFinanceData(financeRes.data[0]);
      }
    } catch (err) {
      console.warn("Fetch finance error", err);
    }
  };

  const handleExportReport = () => {
    if (!selectedRec) return;

    // 生成报告摘要文本
    const reportText = `
【StockSearch 专业个股量化分析报告】
-----------------------------------
股票简称：${individualInfo?.['股票简称'] || '未知'}
股票代码：${selectedRec.symbol}
所属行业：${individualInfo?.['行业'] || '未知'}
操作建议：${selectedRec.action}
综合置信度：${selectedRec.score}

【AI 深度解析】
${selectedRec.advice}

【底层依据】
${selectedRec.reasons?.join(' | ')}

【财务核心指标】
ROE: ${financeData?.['净资产收益率'] || '--'}
PE: ${financeData?.['市盈率'] || '--'}
营收: ${financeData?.['营业收入'] || '--'}

【执行建议】
建议入场价: ${selectedRec.price || '--'}
预期目标位: ${(selectedRec.price * 1.15).toFixed(2)}
风险止损位: ${(selectedRec.price * 0.95).toFixed(2)}

报告生成时间: ${new Date().toLocaleString()}
    `;

    // 复制到剪贴板
    navigator.clipboard.writeText(reportText).then(() => {
      alert("✅ 专业分析简报已复制到剪贴板！\n同时为您开启浏览器打印功能，您可以直接保存为 PDF 报告。");
      window.print();
    }).catch(err => {
      console.error("Copy error", err);
      window.print();
    });
  };

  return (
    <div className="flex h-screen w-full bg-background text-textMain overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-gray-800 bg-card flex flex-col shrink-0 overflow-hidden">
        <div className="p-6 flex items-center gap-3">
          <div className="p-2 bg-primary rounded-xl shadow-lg shadow-primary/20">
            <TrendingUp size={24} className="text-white" />
          </div>
          <h1 className="text-xl font-black tracking-tighter text-white">STOCKSEARCH</h1>
        </div>

        <nav className="flex-1 px-4 space-y-2 overflow-y-auto pt-2">
          <NavItem icon={<LayoutDashboard size={20} />} label="监控概览" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <NavItem icon={<Newspaper size={20} />} label="实时新闻" active={activeTab === 'news'} onClick={() => setActiveTab('news')} />
          <NavItem icon={<History size={20} />} label="算法复盘" active={activeTab === 'review'} onClick={() => setActiveTab('review')} />
          <NavItem icon={<BarChart3 size={20} />} label="全场排行" active={activeTab === 'rank'} onClick={() => setActiveTab('rank')} />
          <NavItem icon={<Settings size={20} />} label="系统设置" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
        </nav>

        <div className="p-4 m-4 bg-gray-900/50 rounded-2xl border border-gray-800">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 bg-success rounded-full animate-pulse" />
            <span className="text-xs font-bold text-gray-400 uppercase">Engine Status</span>
          </div>
          <p className="text-[11px] text-gray-500 leading-tight">量化策略引擎处于活跃状态，自迭代模块监控中...</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0 bg-background overflow-hidden relative">
        {/* Header */}
        <header className="h-20 shrink-0 border-b border-gray-800 bg-card/80 backdrop-blur-md flex items-center justify-between px-8 z-10">
          <div className="relative group w-96">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-500 group-focus-within:text-primary transition-colors">
              <Search size={18} />
            </div>
            <input
              type="text"
              className="w-full bg-gray-900/50 border border-gray-800 rounded-2xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-white placeholder:text-gray-600 shadow-inner"
              placeholder="搜索任何股票代码进行深度诊断..."
              value={searchSymbol}
              onChange={(e) => setSearchSymbol(e.target.value)}
              onKeyDown={handleSearch}
            />
          </div>

          <div className="flex items-center gap-6">
            <button onClick={fetchData} className={`p-2.5 bg-gray-800 hover:bg-gray-700 rounded-xl text-gray-400 hover:text-white transition-all ${loadingNews || loadingRecs ? 'animate-spin' : ''}`}>
              <RefreshCw size={20} />
            </button>
            <div className="h-8 w-[1px] bg-gray-800 mx-2" />
            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-sm font-bold text-white leading-none">Admin User</p>
                <p className="text-[10px] text-primary font-black uppercase tracking-widest mt-1">QUANT MASTER</p>
              </div>
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-blue-600 rounded-2xl flex items-center justify-center font-black text-white shadow-lg">AD</div>
            </div>
          </div>
        </header>

        {/* Content Tabs */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-8">
          {activeTab === 'news' ? (
            <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between">
                <h2 className="text-3xl font-black text-white flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-lg"><Newspaper className="text-primary" size={28} /></div>
                  全市场实时快讯
                </h2>
              </div>
              <div className="grid grid-cols-1 gap-4">
                {news.map((item, idx) => <NewsDetailedCard key={idx} item={item} onShowDetail={setScoreDetail} />)}
              </div>
            </div>
          ) : activeTab === 'review' ? (
            <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-black text-white flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg"><History className="text-primary" size={28} /></div>
                    绩效复盘
                  </h2>
                  <p className="text-gray-500 mt-2 text-sm font-medium">通过历史实盘数据驱动核心权重因子的闭环自完善</p>
                </div>
                <button onClick={handleIterate} className="px-6 py-3 bg-primary hover:bg-primary/80 active:scale-95 text-white rounded-2xl flex items-center gap-2 text-sm font-black shadow-xl shadow-primary/30 transition-all uppercase tracking-widest">
                  <RefreshCw size={18} /> 进化计算单元
                </button>
              </div>

              <BacktestPerformance performance={performance} />
            </div>
          ) : activeTab === 'settings' ? (
            <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <h2 className="text-3xl font-black text-white flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg"><Settings className="text-primary" size={28} /></div>
                系统参数设定
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-card border border-gray-800 rounded-[32px] p-8 space-y-6 shadow-xl">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xl font-black text-white uppercase tracking-wider">策略引擎配置</h3>
                    <button
                      onClick={handleSaveWeights}
                      disabled={savingWeights}
                      className="px-4 py-2 bg-primary/20 hover:bg-primary/40 text-primary text-[10px] font-black uppercase tracking-widest rounded-xl transition-all border border-primary/30"
                    >
                      {savingWeights ? '保存中...' : '提交修改'}
                    </button>
                    <button
                      onClick={handleIterate}
                      className="px-4 py-2 bg-success/20 hover:bg-success/40 text-success text-[10px] font-black uppercase tracking-widest rounded-xl transition-all border border-success/30"
                    >
                      AI自迭代
                    </button>
                  </div>
                  <div className="space-y-8">
                    <div className="p-5 bg-gray-900/50 rounded-2xl border border-gray-800 group">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-xs font-black text-gray-400 uppercase tracking-widest">技术面权重 (Technical)</span>
                        <span className="text-primary font-black text-lg">{(weights.tech_weight * 100).toFixed(0)}%</span>
                      </div>
                      <input
                        type="range" min="0" max="1" step="0.05"
                        value={weights.tech_weight}
                        onChange={(e) => handleWeightChange('tech', e.target.value)}
                        className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-primary"
                      />
                    </div>
                    <div className="p-5 bg-gray-900/50 rounded-2xl border border-gray-800 group">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-xs font-black text-gray-400 uppercase tracking-widest">消息面权重 (Sentiment)</span>
                        <span className="text-success font-black text-lg">{(weights.sentiment_weight * 100).toFixed(0)}%</span>
                      </div>
                      <input
                        type="range" min="0" max="1" step="0.05"
                        value={weights.sentiment_weight}
                        onChange={(e) => handleWeightChange('sent', e.target.value)}
                        className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-success"
                      />
                    </div>
                  </div>
                  <div className="flex items-start gap-2 p-3 bg-warning/5 border border-warning/10 rounded-xl">
                    <AlertCircle size={14} className="text-warning shrink-0 mt-0.5" />
                    <p className="text-[10px] text-gray-500 font-medium leading-relaxed italic">
                      提示：技术面侧重趋势跟踪 (EMA)，消息面侧重实时情绪 (NLP)。拖动滑块后需点击“提交修改”生效。
                    </p>
                  </div>
                </div>

                <div className="bg-card border border-gray-800 rounded-[32px] p-8 space-y-6 shadow-xl">
                  <h3 className="text-xl font-black text-white uppercase tracking-wider">数据源状态</h3>
                  <div className="space-y-4">
                    <StatusItem label="实时行情 (Spot)" status="Online" />
                    <StatusItem label="历史K线 (History)" status="Online" />
                    <StatusItem label="财经电报 (Flash)" status="Online" />
                    <StatusItem label="语义分析 (NLP)" status="Idle" />
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-primary/10 to-transparent border border-primary/20 p-8 rounded-[32px] flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-primary rounded-2xl">
                    <AlertCircle size={24} className="text-white" />
                  </div>
                  <div>
                    <h4 className="font-black text-white text-lg">进阶自迭代模式</h4>
                    <p className="text-gray-400 text-sm">系统当前处于稳健进化模式，每累计 5 笔交易将触发一次深度学习回归</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-8 animate-in fade-in duration-700">
              {/* Stat Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard label="TODAY'S SIGNALS" value={recommendations.length} trend={`${recommendations.filter(r => r.action === 'BUY').length} BUY`} color="primary" />
                <StatCard label="SENTIMENT MOMENTUM" value={marketSentiment.score} trend={marketSentiment.trend} isDown={parseFloat(marketSentiment.score) < 0} color="success" />
                <StatCard label="AI ACCURACY" value="68.5%" trend="+1.2%" color="primary" />
                <StatCard label="STRATEGY INDEX" value="8.4" trend="+0.4" color="success" />
              </div>

              {activeTab === 'rank' ? (
                <div className="space-y-6">
                  {/* 行业分布统计区域 */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="md:col-span-1">
                      <IndustryPieChart data={industryStats} />
                    </div>
                    <div className="md:col-span-2 flex items-center p-8 bg-card border border-gray-800 rounded-[32px] shadow-xl relative overflow-hidden group">
                      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                      <div className="relative z-10 flex flex-col gap-4">
                        <div className="flex items-center gap-3 text-primary">
                          <div className="p-2 bg-primary/10 rounded-xl">
                            <PieChart size={24} />
                          </div>
                          <span className="text-xl font-black uppercase tracking-widest">热点快照</span>
                        </div>
                        <p className="text-gray-400 text-sm leading-relaxed max-w-lg">
                          基于全市场前 <span className="text-white font-bold">200</span> 名强势股的行业分布统计。
                          <br />
                          当前资金抱团最明显的板块是：
                          <span className="text-2xl font-black text-white block mt-2">
                            {industryStats.length > 0 ? industryStats[0].name : '分析中...'}
                            <span className="text-sm font-bold text-primary ml-2">
                              {industryStats.length > 0 ? `(${industryStats[0].value} 支)` : ''}
                            </span>
                          </span>
                        </p>
                      </div>
                    </div>
                  </div>

                  <MarketRankTable
                    data={fullRank}
                    onSelectRec={handleSelectRec}
                    loading={loadingRank}
                    pagination={rankPagination}
                    onPageChange={handlePageChange}
                    filter={rankFilter}
                    onFilterChange={setRankFilter}
                    sort={rankSort}
                    onSortChange={setRankSort}
                    sectors={sectors}
                  />
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
                    {/* Main Chart */}
                    <div className="xl:col-span-3 bg-card border border-gray-800 rounded-[32px] p-8 h-[600px] flex flex-col shadow-2xl relative">
                      <div className="flex items-center justify-between mb-8 z-10">
                        <h3 className="font-black text-xl text-white flex items-center gap-3">
                          <BarChart3 size={24} className="text-primary" />
                          量化技术看板：{klineData[0]?.代码 || 'SH000001'}
                        </h3>
                        <div className="flex gap-2 bg-gray-900 border border-gray-800 p-1.5 rounded-2xl">
                          <span className="px-3 py-1.5 bg-gray-800 rounded-xl text-[10px] text-primary font-black uppercase tracking-widest">Trend Following</span>
                          <span className="px-3 py-1.5 text-[10px] text-gray-600 font-bold">EMA 20/60/120</span>
                        </div>
                      </div>
                      <div ref={chartRef} className="flex-1 min-h-0 w-full" />
                    </div>

                    {/* News Side Scroll */}
                    <div className="bg-card border border-gray-800 rounded-[32px] p-8 flex flex-col h-[600px] shadow-2xl overflow-hidden">
                      <h3 className="font-black text-xl mb-6 flex items-center justify-between text-white">
                        <span className="flex items-center gap-2">
                          <Newspaper size={20} className="text-primary" /> 实时消息面
                        </span>
                        {loadingNews && <RefreshCw size={14} className="animate-spin text-primary" />}
                      </h3>
                      <div className="flex-1 space-y-5 overflow-y-auto pr-2 custom-scrollbar">
                        {news.map((item, idx) => (
                          <NewsItem key={idx} time={item.时间 || item.time} content={item.内容 || item.content} sentiment={item.sentiment} onShowDetail={setScoreDetail} />
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Recommendation Sections */}
                  <div className="space-y-6 mt-8">
                    <div className="flex items-center gap-4">
                      <div className="h-0.5 flex-1 bg-gradient-to-r from-transparent via-gray-800 to-transparent" />
                      <h3 className="font-black text-xl text-white uppercase tracking-[0.2em] whitespace-nowrap">AI Smart Recommends</h3>
                      <div className="h-0.5 flex-1 bg-gradient-to-r from-gray-800 via-gray-800 to-transparent" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      {recommendations.map((rec, id) => <RecCard key={id} rec={rec} onSelect={handleSelectRec} />)}
                      {loadingRecs && [1, 2, 3, 4].map(i => <div key={i} className="bg-card border border-gray-800 rounded-[24px] p-6 h-48 animate-pulse shadow-sm" />)}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Detail Modal Overlay */}
        {selectedRec && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 backdrop-blur-xl bg-black/40 animate-in fade-in duration-300">
            <div className="bg-card border border-gray-800 w-full max-w-6xl max-h-[90vh] rounded-[40px] shadow-[0_0_100px_rgba(0,0,0,0.8)] flex flex-col overflow-hidden relative border-t-primary/20">
              {/* Close Button */}
              <button
                onClick={() => setSelectedRec(null)}
                className="absolute top-6 right-8 p-3 bg-gray-900 border border-gray-800 text-gray-400 hover:text-white rounded-2xl hover:bg-gray-800 transition-all z-20"
              >
                <X size={20} />
              </button>

              <div className="flex-1 overflow-y-auto custom-scrollbar p-10 pt-12">
                <div className="flex items-start justify-between mb-10">
                  <div className="flex items-center gap-6">
                    <div className="w-20 h-20 bg-primary/10 rounded-3xl flex items-center justify-center text-3xl font-black text-primary border border-primary/20">
                      {selectedRec.symbol.slice(0, 2)}
                    </div>
                    <div>
                      <div className="flex items-center gap-3">
                        <h2 className="text-4xl font-black text-white">{individualInfo?.['股票简称'] || selectedRec.symbol}</h2>
                        <span className="text-gray-500 text-xl font-bold bg-gray-900 px-3 py-1 rounded-xl border border-gray-800">{selectedRec.symbol}</span>
                        <span className={`px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-widest ${selectedRec.action === 'BUY' ? 'bg-success text-white' : 'bg-primary text-white'
                          }`}>{selectedRec.action} SIGNAL</span>
                      </div>
                      <div className="flex items-center gap-4 mt-3">
                        <p className="text-gray-400 font-bold flex items-center gap-2 text-sm bg-primary/10 px-3 py-1 rounded-lg border border-primary/20">
                          <BarChart3 size={14} className="text-primary" />
                          {individualInfo?.['行业'] || '行业待核实'}
                        </p>
                        <p className="text-gray-500 font-bold flex items-center gap-2 text-sm">
                          <Info size={14} className="text-primary" />
                          置信度评分 <span className="text-primary">{selectedRec.score}</span>
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-10">
                  {/* Detailed Chart */}
                  <div className="lg:col-span-2 bg-gray-900/50 border border-gray-800 rounded-[32px] p-6 h-[450px] flex flex-col relative overflow-hidden group/chart">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Quantum Technical Engine</span>
                      {loadingChart && <div className="flex gap-1"><div className="w-1 h-1 bg-primary rounded-full animate-bounce" /><div className="w-1 h-1 bg-primary rounded-full animate-bounce [animation-delay:-.15s]" /><div className="w-1 h-1 bg-primary rounded-full animate-bounce [animation-delay:-.3s]" /></div>}
                    </div>
                    <div ref={modalChartRef} className="flex-1 w-full min-h-0" />
                    {!loadingChart && modalData.length === 0 && (
                      <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-xs font-bold uppercase tracking-widest bg-gray-900/50">
                        No K-Line Data Available
                      </div>
                    )}
                  </div>

                  {/* Strategy Analysis */}
                  <div className="space-y-6">
                    <div className="bg-primary/5 border border-primary/10 rounded-[32px] p-6">
                      <h4 className="text-xs font-black text-primary uppercase tracking-widest mb-4 flex items-center gap-2">
                        <TrendingUp size={14} /> AI 决策深度分析
                      </h4>
                      <p className="text-white font-medium leading-[1.8] italic">"{selectedRec.advice}"</p>
                    </div>

                    <div className="bg-card border border-gray-800 rounded-[32px] p-6">
                      <h4 className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-4">底层支撑因子</h4>
                      <div className="space-y-3">
                        {selectedRec.reasons?.map((r, i) => (
                          <div key={i} className="flex items-center gap-3 p-3 bg-gray-900/50 rounded-2xl border border-gray-800 group hover:border-primary/30 transition-all">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_8px_#3b82f6]" />
                            <span className="text-sm font-bold text-gray-300">{r}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Finance Data Grid */}
                {financeData && (
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-10">
                    <FinanceItem label="ROE" value={financeData['净资产收益率'] || '--'} />
                    <FinanceItem label="PE (市盈率)" value={financeData['市盈率'] || '--'} />
                    <FinanceItem label="PB (市净率)" value={financeData['市净率'] || '--'} />
                    <FinanceItem label="营收" value={financeData['营业收入'] || '--'} />
                    <FinanceItem label="净利润" value={financeData['净利润'] || '--'} />
                    <FinanceItem label="毛利率" value={financeData['销售毛利率'] || '--'} />
                  </div>
                )}

                <div className="p-8 bg-gradient-to-br from-card to-gray-900 rounded-[32px] border border-gray-800">
                  <div className="flex items-center justify-between mb-6 print:hidden">
                    <h4 className="text-lg font-black text-white">操作执行建议</h4>
                    <div
                      onClick={handleExportReport}
                      className="flex items-center gap-2 text-primary font-black text-sm cursor-pointer hover:underline"
                    >
                      <ExternalLink size={16} /> 导出专业报告
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-sm">
                    <SummaryItem label="入场触发价" value={selectedRec.price || '--'} sub="实时盯盘" />
                    <SummaryItem label="止盈目标" value={(selectedRec.price * 1.15).toFixed(2)} sub="预估收益 +15%" />
                    <SummaryItem label="止损保护" value={(selectedRec.price * 0.95).toFixed(2)} sub="最大回撤控制 -5%" />
                    <SummaryItem label="上市日期" value={individualInfo?.['上市时间'] || '--'} sub="时空跨度" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 情感得分详情弹窗 (穿透分析) */}
        {scoreDetail && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 animate-in">
            <div className="absolute inset-0 bg-black/80 backdrop-blur-xl" onClick={() => setScoreDetail(null)} />
            <div className="relative w-full max-w-lg bg-card border border-gray-800 rounded-[40px] shadow-[0_0_100px_rgba(0,0,0,0.8)] overflow-hidden">
              <div className="p-8 space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <h3 className="text-2xl font-black text-white">AI 情感价值分析</h3>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 bg-primary/20 text-primary text-[10px] font-black rounded-md border border-primary/20 uppercase tracking-widest">
                        定位行业: {scoreDetail.sector || '全市场'}
                      </span>
                    </div>
                  </div>
                  <button onClick={() => setScoreDetail(null)} className="text-gray-500 hover:text-white transition-colors">
                    <XCircle size={24} />
                  </button>
                </div>

                <div className="bg-gray-900/50 p-6 rounded-3xl border border-white/[0.03] space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400 font-bold uppercase tracking-widest">综合影响得分</span>
                    <span className={`text-4xl font-black ${scoreDetail.score > 0 ? 'text-success' : scoreDetail.score < 0 ? 'text-danger' : 'text-primary'}`}>
                      {scoreDetail.score}
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-1000 ${scoreDetail.score > 0 ? 'bg-success' : scoreDetail.score < 0 ? 'bg-danger' : 'bg-primary'}`}
                      style={{ width: `${Math.abs(scoreDetail.score) * 100}%`, marginLeft: scoreDetail.score < 0 ? '0' : '0' }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <p className="text-[10px] text-success font-black uppercase tracking-widest flex items-center gap-2">
                      <ArrowUpCircle size={12} /> 利好关键词
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {scoreDetail.pos_words?.length > 0 ? scoreDetail.pos_words.map(w => (
                        <span key={w} className="px-3 py-1 bg-success/10 text-success text-xs font-bold rounded-lg border border-success/20">{w}</span>
                      )) : <span className="text-gray-600 text-xs italic">无明显利好</span>}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <p className="text-[10px] text-danger font-black uppercase tracking-widest flex items-center gap-2">
                      <ArrowDownCircle size={12} /> 利空关键词
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {scoreDetail.neg_words?.length > 0 ? scoreDetail.neg_words.map(w => (
                        <span key={w} className="px-3 py-1 bg-danger/10 text-danger text-xs font-bold rounded-lg border border-danger/20">{w}</span>
                      )) : <span className="text-gray-600 text-xs italic">无明显利空</span>}
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-gray-900/30 rounded-2xl border border-white/[0.02]">
                  <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-2 flex items-center gap-2">
                    <Info size={12} /> 诊断逻辑
                  </p>
                  <p className="text-xs text-gray-400 leading-relaxed italic">
                    {scoreDetail.reasoning || "基于关键词特征进行的中性打分。"}
                  </p>
                </div>

                <p className="text-xs text-gray-500 italic leading-relaxed pt-4 border-t border-gray-800">
                  * 评分备注：AI 根据 A 股核心词库，通过情感分量及语义强度自动生成的概率性预测，仅供复盘参考。
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 5px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #334155; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-in { animation: fadeIn 0.4s ease-out forwards; }
        
        @media print {
          body * { visibility: hidden; }
          .fixed, .fixed * { visibility: visible; }
          .fixed { position: absolute; left: 0; top: 0; width: 100%; border: none !important; }
          .print\:hidden { display: none !important; }
          .bg-card, .bg-gray-900\/50, .bg-gray-900\/30 { background: white !important; color: black !important; border-color: #eee !important; }
          .text-white, .text-gray-300, .text-gray-400 { color: black !important; }
          .text-primary { color: #2563eb !important; }
          button, .absolute.top-6.right-8 { display: none !important; }
          .rounded-\[40px\], .rounded-\[32px\] { border-radius: 8px !important; }
          .shadow-\[0_0_100px_rgba\(0,0,0,0.8\)\] { shadow: none !important; }
        }
      `}</style>
    </div >
  );
};

const NavItem = ({ icon, label, active, onClick }) => (
  <button onClick={onClick} className={`w-full group flex items-center gap-4 px-5 py-4 rounded-[20px] transition-all duration-300 relative ${active ? 'bg-primary text-white shadow-[0_10px_30px_-10px_rgba(59,130,246,0.5)]' : 'text-gray-500 hover:bg-gray-800/50 hover:text-gray-300'
    }`}>
    <div className={`transition-transform duration-500 ${active ? 'scale-110' : 'group-hover:scale-110'}`}>{icon}</div>
    <span className="font-black text-sm tracking-wide">{label}</span>
    {active && <div className="absolute right-4 w-1.5 h-1.5 bg-white rounded-full shadow-[0_0_10px_white]" />}
  </button>
);

const StatCard = ({ label, value, trend, isDown, color }) => (
  <div className="bg-card border border-gray-800 rounded-[28px] p-7 group hover:border-primary/50 transition-all shadow-xl hover:-translate-y-1 duration-300 relative overflow-hidden">
    <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary/5 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity" />
    <p className="text-gray-500 text-[10px] font-black mb-3 uppercase tracking-widest leading-none">{label}</p>
    <div className="flex items-end justify-between relative z-10">
      <h3 className="text-3xl font-black text-white tracking-tighter">{value}</h3>
      <div className={`flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-[10px] font-black ${isDown ? 'bg-danger/10 text-danger' : 'bg-success/10 text-success'}`}>
        {trend}
      </div>
    </div>
  </div>
);

const NewsItem = ({ time, content, sentiment, onShowDetail }) => {
  const isPos = sentiment?.sentiment === 'positive';
  const isNeg = sentiment?.sentiment === 'negative';
  return (
    <div className={`p-4 rounded-2xl border border-gray-800/50 bg-gray-900/20 hover:bg-gray-800/40 transition-all group cursor-default relative overflow-hidden`}>
      {isPos && <div className="absolute left-0 top-0 bottom-0 w-1 bg-success/40" />}
      {isNeg && <div className="absolute left-0 top-0 bottom-0 w-1 bg-danger/40" />}
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-500">
            {(() => {
              const timeStr = time || '';
              // 如果是完整日期时间格式（包含日期），格式化显示
              if (timeStr.includes('-') && timeStr.includes(':')) {
                try {
                  const date = new Date(timeStr);
                  const today = new Date();
                  const isToday = date.toDateString() === today.toDateString();
                  if (isToday) {
                    // 今天的新闻只显示时间
                    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
                  } else {
                    // 非今天的新闻显示日期和时间
                    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
                  }
                } catch {
                  return timeStr;
                }
              }
              return timeStr;
            })()}
          </span>
          {sentiment?.sector && <span className="text-[9px] font-black text-primary/60 uppercase">{sentiment.sector}</span>}
        </div>
        {sentiment && (
          <span
            onClick={(e) => { e.stopPropagation(); onShowDetail(sentiment); }}
            className={`text-[8px] font-black px-1.5 py-0.5 rounded uppercase border cursor-help hover:scale-105 transition-transform ${isPos ? 'text-success border-success/30 bg-success/5' : isNeg ? 'text-danger border-danger/30 bg-danger/5' : 'text-gray-500 border-gray-700'}`}
          >
            Score: {sentiment.score}
          </span>
        )}
      </div>
      <p className="text-xs text-gray-400 leading-relaxed font-medium line-clamp-2 group-hover:line-clamp-none transition-all duration-500">{content}</p>
    </div>
  );
};

const NewsDetailedCard = ({ item, onShowDetail }) => {
  const isPos = item.sentiment?.sentiment === 'positive';
  const isNeg = item.sentiment?.sentiment === 'negative';
  return (
    <div className={`bg-card border p-8 rounded-[32px] transition-all hover:shadow-2xl ${isPos ? 'border-success/20 bg-gradient-to-br from-card to-success/5' :
      isNeg ? 'border-danger/20 bg-gradient-to-br from-card to-danger/5' : 'border-gray-800'
      }`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isPos ? 'bg-success shadow-[0_0_10px_#10b981]' : isNeg ? 'bg-danger shadow-[0_0_10px_#ef4444]' : 'bg-gray-600'}`} />
          <span className="text-sm font-black text-gray-400 font-mono italic">
            {(() => {
              const timeStr = item.时间 || item.time || '';
              const created_at = item.created_at || '';
              
              // 如果时间只有时间没有日期，尝试使用 created_at 补充日期
              let fullTimeStr = timeStr;
              if (timeStr && !timeStr.includes('-') && timeStr.includes(':')) {
                // 只有时间，没有日期（格式如 "19:34:12"）
                if (created_at) {
                  try {
                    const createdDate = new Date(created_at);
                    if (!isNaN(createdDate.getTime())) {
                      fullTimeStr = `${createdDate.getFullYear()}-${String(createdDate.getMonth() + 1).padStart(2, '0')}-${String(createdDate.getDate()).padStart(2, '0')} ${timeStr}`;
                    } else {
                      // created_at 解析失败，使用当前日期
                      const now = new Date();
                      fullTimeStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${timeStr}`;
                    }
                  } catch (e) {
                    // 如果 created_at 解析失败，使用当前日期
                    const now = new Date();
                    fullTimeStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${timeStr}`;
                  }
                } else {
                  // 没有 created_at，使用当前日期
                  const now = new Date();
                  fullTimeStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${timeStr}`;
                }
              }
              
              // 如果是完整日期时间格式（包含日期），格式化显示
              if (fullTimeStr.includes('-') && fullTimeStr.includes(':')) {
                try {
                  const date = new Date(fullTimeStr);
                  if (isNaN(date.getTime())) {
                    return timeStr; // 解析失败，返回原始值
                  }
                  const today = new Date();
                  today.setHours(0, 0, 0, 0);
                  const newsDate = new Date(date);
                  newsDate.setHours(0, 0, 0, 0);
                  const isToday = newsDate.getTime() === today.getTime();
                  
                  if (isToday) {
                    // 今天的新闻只显示时间
                    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
                  } else {
                    // 非今天的新闻显示日期和时间
                    return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
                  }
                } catch (e) {
                  return timeStr;
                }
              }
              return timeStr;
            })()}
          </span>
          {item.sentiment?.sector && (
            <span className="px-2 py-0.5 bg-gray-800 text-gray-500 text-[10px] font-black rounded border border-gray-700 uppercase tracking-widest">
              {item.sentiment.sector}
            </span>
          )}
        </div>
        {item.sentiment && (
          <div
            onClick={(e) => { e.stopPropagation(); onShowDetail(item.sentiment); }}
            className={`px-4 py-1.5 rounded-full text-xs font-black uppercase tracking-widest cursor-help hover:brightness-125 transition-all ${isPos ? 'bg-success text-white shadow-lg shadow-success/20' : isNeg ? 'bg-danger text-white shadow-lg shadow-danger/20' : 'bg-gray-800 text-gray-400'
              }`}>IMPACT SCORE: {item.sentiment.score}</div>
        )}
      </div>
      <p className="text-lg text-white font-medium leading-[1.6] leading-relaxed">{item.内容 || item.content}</p>
    </div>
  );
};

const RecCard = ({ rec, onSelect }) => {
  const signalStyle = {
    BUY: 'bg-success text-white shadow-lg shadow-success/30',
    HOLD: 'bg-primary text-white shadow-lg shadow-primary/30',
    SELL: 'bg-danger text-white shadow-lg shadow-danger/30',
    WAIT: 'bg-gray-700 text-gray-300'
  }[rec.action] || 'bg-gray-800 text-gray-400';

  return (
    <div
      onClick={() => onSelect(rec)}
      className="bg-card border border-gray-800 rounded-[28px] p-6 hover:border-primary transition-all duration-500 relative group overflow-hidden shadow-xl hover:-translate-y-2 cursor-pointer active:scale-95"
    >
      <div className={`absolute top-0 right-0 px-5 py-2 text-[10px] font-black uppercase tracking-[0.2em] rounded-bl-2xl ${signalStyle}`}>
        {rec.action}
      </div>

      <div className="flex items-center gap-4 mb-5">
        <div className="w-14 h-14 bg-gray-900 border border-gray-800 rounded-2xl flex items-center justify-center font-black text-2xl text-primary group-hover:text-white group-hover:bg-primary transition-all duration-500 shadow-inner">
          {(rec.name || rec.symbol || "?").slice(0, 1)}
        </div>
        <div>
          <h4 className="font-black text-xl text-white tracking-tight line-clamp-1">{rec.name || rec.symbol}</h4>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{rec.symbol}:</span>
            <span className="text-sm font-black text-primary">{rec.score}</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-900/50 rounded-2xl p-4 mb-4 border border-white/[0.03]">
        <p className="text-[11px] text-gray-300 font-bold leading-relaxed italic line-clamp-2">"{rec.advice}"</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {rec.reasons?.map((r, i) => (
          <span key={i} className="text-[9px] font-black uppercase bg-white/[0.03] text-gray-500 px-2.5 py-1.5 rounded-xl border border-white/[0.05] group-hover:border-primary/20 hover:text-gray-300 transition-colors">
            {r}
          </span>
        ))}
      </div>
    </div>
  );
};

const MarketRankTable = ({ data, onSelectRec, loading, pagination, onPageChange, filter, onFilterChange, sort, onSortChange, sectors = [] }) => {
  // 移除本地状态和 useMemo，改为使用 props

  const handleSort = (key) => {
    let direction = 'desc';
    if (sort.key === key && sort.direction === 'desc') {
      direction = 'asc';
    }
    onSortChange({ key, direction });
  };

  // 移除本地 filteredData，数据已经是过滤后的


  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-wrap items-center gap-4 bg-card border border-gray-800 p-6 rounded-[32px] shadow-xl">
        <div className="flex-1 min-w-[200px] relative group">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-primary" size={18} />
          <input
            type="text" placeholder="搜索名称/代码..."
            className="w-full bg-gray-900/50 border border-gray-800 rounded-2xl py-3 pl-12 pr-4 text-sm focus:ring-2 focus:ring-primary/20 text-white"
            value={filter.search} onChange={e => onFilterChange({ ...filter, search: e.target.value })}
          />
        </div>
        <select
          className="bg-gray-900/50 border border-gray-800 rounded-2xl py-3 px-6 text-sm text-white focus:ring-2 focus:ring-primary/20 outline-none"
          value={filter.sector} onChange={e => onFilterChange({ ...filter, sector: e.target.value })}
        >
          <option value="">全行业</option>
          {sectors.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <div className="flex items-center gap-2">
          <input
            type="number" placeholder="最低价"
            className="w-24 bg-gray-900/50 border border-gray-800 rounded-2xl py-3 px-4 text-sm text-white"
            value={filter.minPrice} onChange={e => onFilterChange({ ...filter, minPrice: e.target.value })}
          />
          <span className="text-gray-600">-</span>
          <input
            type="number" placeholder="最高价"
            className="w-24 bg-gray-900/50 border border-gray-800 rounded-2xl py-3 px-4 text-sm text-white"
            value={filter.maxPrice} onChange={e => onFilterChange({ ...filter, maxPrice: e.target.value })}
          />
        </div>
      </div>

      <div className="bg-card border border-gray-800 rounded-[32px] overflow-hidden shadow-2xl relative min-h-[400px]">
        {loading && (
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm z-10 flex items-center justify-center">
            <RefreshCw className="animate-spin text-primary" size={40} />
          </div>
        )}
        <table className="w-full text-left">
          <thead className="bg-gray-900/50">
            <tr className="text-gray-400 text-[11px] font-black uppercase tracking-widest border-b border-gray-800">
              <th className="px-8 py-6 cursor-pointer hover:text-white" onClick={() => handleSort('symbol')}>
                <div className="flex items-center gap-2">
                  Symbol / Name
                  {sort.key === 'symbol' && (sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </div>
              </th>
              <th className="px-8 py-6 cursor-pointer hover:text-white" onClick={() => handleSort('industry')}>
                <div className="flex items-center gap-2">
                  Sector
                  {sort.key === 'industry' && (sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </div>
              </th>
              <th className="px-8 py-6 text-center cursor-pointer hover:text-white" onClick={() => handleSort('price')}>
                <div className="flex items-center justify-center gap-2">
                  Price
                  {sort.key === 'price' && (sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </div>
              </th>
              <th className="px-8 py-6 text-center cursor-pointer hover:text-white" onClick={() => handleSort('change')}>
                <div className="flex items-center justify-center gap-2">
                  Daily Change
                  {sort.key === 'change' && (sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </div>
              </th>
              <th className="px-8 py-6 text-center cursor-pointer hover:text-white" onClick={() => handleSort('score')}>
                <div className="flex items-center justify-center gap-2">
                  Score
                  {sort.key === 'score' && (sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </div>
              </th>
              <th className="px-8 py-6 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {data.map((item, idx) => (
              <tr key={idx} className="hover:bg-white/[0.02] transition-colors group cursor-pointer" onClick={() => onSelectRec(item)}>
                <td className="px-8 py-6">
                  <div className="font-black text-white text-lg">{item.name}</div>
                  <div className="text-[10px] text-gray-500 font-mono">{item.symbol}</div>
                </td>
                <td className="px-8 py-6">
                  <span className="px-3 py-1 bg-gray-900 border border-gray-800 text-gray-400 text-[10px] font-black rounded-lg uppercase">
                    {item.industry || 'N/A'}
                  </span>
                </td>
                <td className="px-8 py-6 text-center font-bold text-white">{item.price?.toFixed(2)}</td>
                <td className="px-8 py-6 text-center">
                  <span className={`text-sm font-black ${item.change >= 0 ? 'text-success' : 'text-danger'}`}>
                    {item.change > 0 ? '+' : ''}{item.change?.toFixed(2)}%
                  </span>
                </td>
                <td className="px-8 py-6 text-center">
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-primary/10 rounded-2xl border border-primary/20 text-primary font-black text-lg">
                    {item.score}
                  </div>
                </td>
                <td className="px-8 py-6 text-right">
                  <span className={`px-4 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border ${item.action === 'BUY' ? 'bg-success text-white border-success/20' : 'bg-primary text-white border-primary/20'}`}>
                    {item.action}
                  </span>
                </td>
              </tr>
            ))}
            {!data.length && !loading && (
              <tr><td colSpan="6" className="px-8 py-20 text-center text-gray-600 italic font-medium">未找到符合条件的个股数据</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* 分页控件 */}
      {pagination && pagination.total_pages > 1 && (
        <div className="flex items-center justify-between mt-6 px-8">
          <div className="text-sm text-gray-400">
            显示第 <span className="font-bold text-white">{((pagination.page - 1) * pagination.page_size) + 1}</span> 至{' '}
            <span className="font-bold text-white">{Math.min(pagination.page * pagination.page_size, pagination.total)}</span> 条，
            共 <span className="font-bold text-primary">{pagination.total}</span> 支股票
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => onPageChange(pagination.page - 1)}
              disabled={pagination.page <= 1 || loading}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-xl transition-all text-sm font-bold"
            >
              上一页
            </button>

            <div className="flex items-center gap-2">
              {/* 页码显示 */}
              {(() => {
                const { page, total_pages } = pagination;
                const pages = [];
                const maxVisible = 7;

                if (total_pages <= maxVisible) {
                  for (let i = 1; i <= total_pages; i++) pages.push(i);
                } else {
                  if (page <= 4) {
                    for (let i = 1; i <= 5; i++) pages.push(i);
                    pages.push('...');
                    pages.push(total_pages);
                  } else if (page >= total_pages - 3) {
                    pages.push(1);
                    pages.push('...');
                    for (let i = total_pages - 4; i <= total_pages; i++) pages.push(i);
                  } else {
                    pages.push(1);
                    pages.push('...');
                    for (let i = page - 1; i <= page + 1; i++) pages.push(i);
                    pages.push('...');
                    pages.push(total_pages);
                  }
                }

                return pages.map((p, idx) => {
                  if (p === '...') {
                    return <span key={`ellipsis-${idx}`} className="text-gray-600 px-2">...</span>;
                  }
                  return (
                    <button
                      key={p}
                      onClick={() => onPageChange(p)}
                      disabled={loading}
                      className={`w-10 h-10 rounded-xl text-sm font-black transition-all ${p === page
                        ? 'bg-primary text-white shadow-lg shadow-primary/30'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'
                        } disabled:opacity-30`}
                    >
                      {p}
                    </button>
                  );
                });
              })()}
            </div>

            <button
              onClick={() => onPageChange(pagination.page + 1)}
              disabled={pagination.page >= pagination.total_pages || loading}
              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-xl transition-all text-sm font-bold"
            >
              下一页
            </button>
          </div>

          <div className="text-sm text-gray-400">
            第 <span className="font-bold text-primary">{pagination.page}</span> / {pagination.total_pages} 页
          </div>
        </div>
      )}
    </div>
  );
};

const StatusItem = ({ label, status }) => (
  <div className="flex items-center justify-between p-3 bg-gray-900/30 rounded-xl border border-white/[0.02]">
    <span className="text-sm text-gray-400 font-bold">{label}</span>
    <div className="flex items-center gap-2">
      <div className={`w-1.5 h-1.5 rounded-full ${status === 'Online' ? 'bg-success shadow-[0_0_5px_#10b981]' : 'bg-gray-600'}`} />
      <span className={`text-[10px] font-black uppercase tracking-widest ${status === 'Online' ? 'text-success' : 'text-gray-600'}`}>{status}</span>
    </div>
  </div>
);

const SummaryItem = ({ label, value, sub }) => (
  <div className="p-5 bg-gray-900/30 rounded-2xl border border-white/[0.03] space-y-2">
    <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{label}</p>
    <p className="text-2xl font-black text-white">{value}</p>
    <p className="text-[10px] text-primary font-medium">{sub}</p>
  </div>
);

const FinanceItem = ({ label, value }) => (
  <div className="p-4 bg-gray-900/50 rounded-2xl border border-white/[0.02] text-center">
    <p className="text-[10px] text-gray-500 font-black mb-1 uppercase tracking-tighter">{label}</p>
    <p className="text-sm font-black text-white">{value}</p>
  </div>
);

export default App;
