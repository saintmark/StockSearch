# StockSearch

> 智能股票推荐系统 - 基于多因子量化分析和新闻情感分析的A股推荐引擎

## 快速开始

### 本地开发

```bash
# 启动后端
cd server
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py

# 启动前端
cd frontend
npm install
npm run dev
```

### Railway 部署

详细部署指南请参考：[Railway 部署文档](readme/RAILWAY_DEPLOYMENT.md)

## 项目结构
 - 智能股票推荐系统

基于多因子量化模型的股票推荐系统，集成LLM新闻情感分析和行业匹配功能。

## 项目结构

```
StockSearch/
├── server/              # 后端服务
│   ├── main.py         # FastAPI主应用
│   ├── strategy_engine.py  # 策略引擎（多因子选股模型）
│   ├── data_fetcher.py      # 数据获取模块
│   ├── sentiment_analyzer.py # 情感分析模块
│   ├── industry_matcher.py   # 行业匹配模块
│   ├── backtest_evaluator.py # 回测评估模块
│   └── ...
├── frontend/           # 前端应用（React）
│   └── src/
│       ├── App.jsx     # 主应用组件
│       └── components/ # 组件目录
├── test/               # 测试脚本目录
│   ├── test_industry_match.py
│   ├── test_integration.py
│   ├── test_strategy_improvements.py
│   └── cleanup_old_data.py
└── readme/             # 项目文档目录
    ├── ALGORITHM_OPTIMIZATION_RECOMMENDATIONS.md
    ├── FEASIBILITY_ANALYSIS.md
    ├── TESTING_GUIDE.md
    └── ...
```

## 快速开始

### 1. 启动后端服务

```bash
cd server
# 激活虚拟环境（Windows）
venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
# 启动服务
python main.py
```

### 2. 启动前端服务

```bash
cd frontend
npm install
npm run dev
```

### 3. 访问应用

打开浏览器访问：http://localhost:5173

## 核心功能

### 1. 多因子选股模型
- **技术因子**：SMA、MACD、RSI、KDJ、OBV、成交量指标
- **基本面因子**：PE、PB、ROE、毛利率
- **风险因子**：波动率、最大回撤、当前回撤
- **消息面因子**：LLM新闻情感分析

### 2. 智能行业匹配
- LLM分析的新闻行业与股票行业的智能匹配
- 多层级匹配策略（直接匹配、反向匹配、关键词匹配）

### 3. 新闻时效性处理
- 时间衰减机制（3个交易日衰减）
- 自动清理过期新闻（7天）

### 4. 回测评估系统
- 固定持仓期评估（5/10/20/30天）
- 动态止盈止损（+10%/-5%）
- 绩效指标计算（胜率、夏普比率、最大回撤等）

## 文档

详细文档请查看 [readme/](readme/) 目录：

- [算法优化建议](readme/ALGORITHM_OPTIMIZATION_RECOMMENDATIONS.md)
- [可行性分析](readme/FEASIBILITY_ANALYSIS.md)
- [测试指南](readme/TESTING_GUIDE.md)
- [行业匹配说明](readme/INDUSTRY_MATCHING_README.md)

## 测试

运行测试脚本：

```bash
cd test
python test_strategy_improvements.py
python test_industry_match.py
python test_integration.py
```

## 许可证

MIT License



