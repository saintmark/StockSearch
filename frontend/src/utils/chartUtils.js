import * as echarts from 'echarts';

export const initStockChart = (domElement, data) => {
    if (!domElement || !data || data.length === 0) return null;

    // 先清理可能存在的实例，防止重复初始化报错
    let chart = echarts.getInstanceByDom(domElement);
    if (chart) {
        chart.dispose();
    }
    chart = echarts.init(domElement, 'dark');

    // 增强字段兼容性：自动尝试中英文 Key
    const dates = data.map(item => item.日期 || item.date || item.Time || "");
    const values = data.map(item => [
        parseFloat(item.开盘 || item.open || item.Open || 0),
        parseFloat(item.收盘 || item.收盘价 || item.close || item.Close || 0),
        parseFloat(item.最低 || item.最低价 || item.low || item.Low || 0),
        parseFloat(item.最高 || item.最高价 || item.high || item.High || 0)
    ]);

    const option = {
        backgroundColor: 'transparent',
        animation: true,
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross', lineStyle: { color: '#3b82f6', width: 1 } },
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            borderColor: '#1e293b',
            textStyle: { color: '#f8fafc' }
        },
        grid: {
            left: '3%',
            right: '3%',
            bottom: '12%',
            top: '5%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: dates,
            scale: true,
            boundaryGap: false,
            axisLine: { lineStyle: { color: '#334155' } },
            splitLine: { show: false },
            axisLabel: { color: '#64748b', fontSize: 10 }
        },
        yAxis: {
            scale: true,
            splitLine: { lineStyle: { color: '#1e293b' } },
            axisLabel: { color: '#64748b', fontSize: 10 }
        },
        dataZoom: [
            { type: 'inside', start: 50, end: 100 },
            { type: 'slider', show: false, start: 50, end: 100 }
        ],
        series: [
            {
                name: '行情',
                type: 'candlestick',
                data: values,
                itemStyle: {
                    color: '#ef4444',     // 涨色 (A 股红)
                    color0: '#10b981',    // 跌色 (A 股绿)
                    borderColor: '#ef4444',
                    borderColor0: '#10b981'
                }
            }
        ]
    };

    chart.setOption(option);

    // 监听窗口缩放以响应式调整
    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);

    // 返回销毁函数或 chart 对象
    return chart;
};
