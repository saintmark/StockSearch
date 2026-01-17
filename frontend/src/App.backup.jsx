import React, { useState, useEffect } from 'react';
import { LayoutDashboard, Newspaper, TrendingUp, History, Settings, Search, Bell } from 'lucide-react';

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="min-h-screen bg-background text-textMain flex overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-card border-r border-gray-800 flex flex-col">
        <div className="p-6 flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <TrendingUp size={20} className="text-white" />
          </div>
          <span className="font-bold text-xl tracking-tight">StockSearch</span>
        </div>

        <nav className="flex-1 px-4 py-4 space-y-2">
          <NavItem
            icon={<LayoutDashboard size={20} />}
            label="监控概览"
            active={activeTab === 'dashboard'}
            onClick={() => setActiveTab('dashboard')}
          />
          <NavItem
            icon={<Newspaper size={20} />}
            label="实时新闻"
            active={activeTab === 'news'}
            onClick={() => setActiveTab('news')}
          />
          <NavItem
            icon={<History size={20} />}
            label="算法复盘"
            active={activeTab === 'review'}
            onClick={() => setActiveTab('review')}
          />
          <NavItem
            icon={<Settings size={20} />}
            label="系统设置"
            active={activeTab === 'settings'}
            onClick={() => setActiveTab('settings')}
          />
        </nav>

        <div className="p-4 border-t border-gray-800 text-xs text-textMuted text-center">
          V1.0.0 (稳健进化版)
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-card border-b border-gray-800 flex items-center justify-between px-8">
          <div className="relative w-96">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-textMuted" size={18} />
            <input
              type="text"
              placeholder="搜索股票代码或名称..."
              className="w-full bg-background border border-gray-700 rounded-full py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-primary transition-colors"
            />
          </div>

          <div className="flex items-center gap-4">
            <button className="p-2 text-textMuted hover:text-textMain hover:bg-gray-800 rounded-full transition-all relative">
              <Bell size={20} />
              <span className="absolute top-1 right-1 w-2 h-2 bg-danger rounded-full border-2 border-card"></span>
            </button>
            <div className="flex items-center gap-3 pl-4 border-l border-gray-800">
              <div className="w-8 h-8 bg-gray-700 rounded-full"></div>
              <span className="text-sm font-medium">管理员</span>
            </div>
          </div>
        </header>

        {/* Dash Content */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard label="今日推荐" value="8" trend="+2" />
            <StatCard label="策略胜率" value="68.5%" trend="+1.2%" />
            <StatCard label="模拟资产" value="￥1,240,500" trend="+0.4%" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
            <div className="xl:col-span-3 bg-card border border-gray-800 rounded-2xl p-6 h-[500px] flex items-center justify-center text-textMuted italic">
              K 线图表加载中 (ECharts 占位)...
            </div>

            <div className="bg-card border border-gray-800 rounded-2xl p-6 flex flex-col">
              <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
                <Newspaper size={18} className="text-primary" />
                最新快讯
              </h3>
              <div className="flex-1 space-y-4 overflow-y-auto pr-2">
                <NewsItem time="16:45" content="多部委发文支持高新区发展，新质生产力板块异动。" />
                <NewsItem time="16:30" content="北向资金今日大幅流出，主要卖出白酒及新能源标的。" />
                <NewsItem time="16:10" content="半导体设备行业再提速，国产替代空间进一步打开。" />
                <NewsItem time="15:45" content="A 股收盘：指教探底回升，科技股表现活跃。" />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

const NavItem = ({ icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${active ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-textMuted hover:bg-gray-800 hover:text-textMain'
      }`}
  >
    {icon}
    <span className="font-medium">{label}</span>
  </button>
);

const StatCard = ({ label, value, trend }) => (
  <div className="bg-card border border-gray-800 rounded-2xl p-6 hover:border-primary/50 transition-colors">
    <p className="text-textMuted text-sm font-medium mb-1">{label}</p>
    <div className="flex items-end justify-between">
      <h3 className="text-2xl font-bold">{value}</h3>
      <span className="text-success text-sm font-bold bg-success/10 px-2 py-0.5 rounded leading-none">
        {trend}
      </span>
    </div>
  </div>
);

const NewsItem = ({ time, content }) => (
  <div className="border-l-2 border-primary/30 pl-3 py-1 hover:border-primary transition-colors">
    <span className="text-xs text-textMuted block mb-1">{time}</span>
    <p className="text-sm leading-relaxed text-textMain line-clamp-2">{content}</p>
  </div>
);

export default App;
