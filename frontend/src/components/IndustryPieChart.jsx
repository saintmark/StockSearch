import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const IndustryPieChart = ({ data }) => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);

    useEffect(() => {
        if (!chartRef.current || !data || data.length === 0) return;

        // 如果实例已存在，先销毁（确保重新渲染）或者直接 setOption
        if (chartInstance.current) {
            chartInstance.current.dispose();
        }

        chartInstance.current = echarts.init(chartRef.current, 'dark', {
            renderer: 'canvas',
        });

        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c} ({d}%)',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: '#1e293b',
                textStyle: { color: '#f8fafc' }
            },
            legend: {
                orient: 'vertical',
                right: 10,
                top: 'center',
                textStyle: { color: '#94a3b8' },
                itemWidth: 10,
                itemHeight: 10
            },
            series: [
                {
                    name: '行业分布',
                    type: 'pie',
                    radius: ['50%', '70%'], // 环形图
                    center: ['35%', '50%'], // 让饼图靠左，留出右边给 Legend
                    avoidLabelOverlap: false,
                    itemStyle: {
                        borderRadius: 10,
                        borderColor: '#0f172a',
                        borderWidth: 2
                    },
                    label: {
                        show: false,
                        position: 'center'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            fontSize: 16,
                            fontWeight: 'bold',
                            color: '#fff'
                        },
                        scale: true,
                        scaleSize: 10
                    },
                    labelLine: {
                        show: false
                    },
                    data: data
                }
            ]
        };

        chartInstance.current.setOption(option);

        const handleResize = () => {
            chartInstance.current && chartInstance.current.resize();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chartInstance.current && chartInstance.current.dispose();
            chartInstance.current = null;
        };
    }, [data]);

    return (
        <div className="bg-card border border-gray-800 rounded-[32px] p-6 shadow-xl h-[300px] flex flex-col">
            <h3 className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-4">Top 200 行业热度分布</h3>
            <div ref={chartRef} className="flex-1 w-full h-full" />
        </div>
    );
};

export default IndustryPieChart;
