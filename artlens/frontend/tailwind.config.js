export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ai:    { DEFAULT: '#dc2626', light: '#fef2f2', border: '#fca5a5' },
        human: { DEFAULT: '#16a34a', light: '#f0fdf4', border: '#86efac' },
        unknown: { DEFAULT: '#d97706', light: '#fffbeb', border: '#fcd34d' },
      }
      ,
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        drift: {
          '0%': { transform: 'translate3d(0, 0, 0) scale(1)' },
          '50%': { transform: 'translate3d(20px, -15px, 0) scale(1.05)' },
          '100%': { transform: 'translate3d(0, 0, 0) scale(1)' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.45' },
          '50%': { opacity: '0.85' },
        },
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        drift: 'drift 14s ease-in-out infinite',
        pulseGlow: 'pulseGlow 4s ease-in-out infinite',
      },
    }
  },
  plugins: [],
}