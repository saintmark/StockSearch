/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0c',
        card: '#15161a',
        primary: '#3b82f6',    // 蓝色 (主色调)
        success: '#10b981',    // 涨 (绿)
        danger: '#ef4444',     // 跌 (红)
        warning: '#f59e0b',    // 预警 (金/黄)
        textMain: '#f1f5f9',
        textMuted: '#94a3b8',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      }
    },
  },
  plugins: [],
}
