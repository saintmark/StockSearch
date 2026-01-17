import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const IndustrySentimentChart = ({ data, weekLabel, marketSentiment }) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!chartRef.current) return;
    
    // 如果没有数据，显示空状态
    if (!data || data.length === 0) {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
      return;
    }

    // 初始化图表
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    // 过滤掉"全市场"，因为全市场单独显示
    const filteredData = data.filter(item => item.industry !== '全市场');
    
    // 按情绪值排序（降序）
    const sortedData = [...filteredData].sort((a, b) => b.sentiment_score - a.sentiment_score);
    
    // 取前15个行业
    const topIndustries = sortedData.slice(0, 15);
    
    const industries = topIndustries.map(item => item.industry);
    const scores = topIndustries.map(item => item.sentiment_score);
    const colors = scores.map(score => {
      if (score >= 1.2) return '#10b981'; // 绿色 - 非常积极
      if (score >= 1.0) return '#84cc16'; // 浅绿 - 积极
      if (score >= 0.8) return '#eab308'; // 黄色 - 中性偏积极
      if (score >= 0.6) return '#f59e0b'; // 橙色 - 中性偏消极
      return '#ef4444'; // 红色 - 消极
    });

    const option = {
      title: {
        text: weekLabel,
        left: 'center',
        textStyle: {
          color: '#ffffff',
          fontSize: 16,
          fontWeight: 'bold'
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: (params) => {
          const param = params[0];
          const dataIndex = param.dataIndex;
          const industry = industries[dataIndex];
          const score = scores[dataIndex];
          const newsCount = topIndustries[dataIndex].news_count || 0;
          return `
            <div style="padding: 8px;">
              <div style="font-weight: bold; margin-bottom: 4px;">${industry}</div>
              <div>情绪值: <span style="color: ${colors[dataIndex]}">${score.toFixed(2)}</span></div>
              <div>新闻数量: ${newsCount}</div>
            </div>
          `;
        }
      },
      grid: {
        left: '15%',
        right: '5%',
        top: '15%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: '情绪值',
        nameTextStyle: {
          color: '#9ca3af'
        },
        axisLabel: {
          color: '#9ca3af'
        },
        axisLine: {
          lineStyle: {
            color: '#374151'
          }
        },
        splitLine: {
          lineStyle: {
            color: '#374151',
            type: 'dashed'
          }
        }
      },
      yAxis: {
        type: 'category',
        data: industries,
        axisLabel: {
          color: '#9ca3af',
          fontSize: 11
        },
        axisLine: {
          lineStyle: {
            color: '#374151'
          }
        }
      },
      series: [
        {
          name: '情绪值',
          type: 'bar',
          data: scores.map((score, index) => ({
            value: score,
            itemStyle: {
              color: colors[index]
            }
          })),
          label: {
            show: true,
            position: 'right',
            formatter: (params) => {
              return typeof params.value === 'number' ? params.value.toFixed(2) : params.value;
            },
            color: '#ffffff',
            fontSize: 11
          },
          barWidth: '60%'
        }
      ]
    };

    chartInstance.current.setOption(option);

    // 响应式调整
    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, weekLabel]);

  return (
    <div className="relative">
      <div ref={chartRef} style={{ width: '100%', height: '500px' }} />
      {marketSentiment !== undefined && (
        <div className="absolute top-4 right-4 bg-card border border-gray-800 rounded-lg p-3 shadow-lg">
          <div className="text-xs text-gray-400 mb-1">全市场情绪值</div>
          <div className={`text-2xl font-bold ${
            marketSentiment >= 1.2 ? 'text-green-500' :
            marketSentiment >= 1.0 ? 'text-lime-500' :
            marketSentiment >= 0.8 ? 'text-yellow-500' :
            marketSentiment >= 0.6 ? 'text-orange-500' : 'text-red-500'
          }`}>
            {marketSentiment.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
};

export default IndustrySentimentChart;

