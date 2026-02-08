/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // LUXURY MEDICAL AI - Premium Color System

        // Base surfaces
        surface: {
          DEFAULT: '#0A0A0F',      // Deep space black
          elevated: '#12121A',     // Card backgrounds
          glass: 'rgba(18, 18, 26, 0.7)',  // Glass overlays
        },

        // Primary brand - Rich Teal/Emerald
        primary: {
          50: '#E6FFF9',
          100: '#B3FFE8',
          200: '#66FFD4',
          300: '#33FFCC',
          400: '#00E6B8',
          500: '#00D4AA',      // Primary brand
          600: '#00B894',
          700: '#009F80',
          800: '#00856A',
          900: '#006B55',
        },

        // Accent - Electric Purple/Violet
        accent: {
          50: '#F3E8FF',
          100: '#E4CCFF',
          200: '#D4B3FF',
          300: '#C299FF',
          400: '#A855F7',      // Vibrant accent
          500: '#9333EA',
          600: '#7C3AED',
          700: '#6D28D9',
          800: '#5B21B6',
          900: '#4C1D95',
        },

        // Medical AI Blue
        ai: {
          50: '#EFF6FF',
          100: '#DBEAFE',
          200: '#BFDBFE',
          300: '#93C5FD',
          400: '#60A5FA',
          500: '#3B82F6',      // AI features
          600: '#2563EB',
          700: '#1D4ED8',
          800: '#1E40AF',
          900: '#1E3A8A',
        },

        // Text hierarchy
        'text-primary': '#FFFFFF',
        'text-secondary': 'rgba(255, 255, 255, 0.7)',
        'text-tertiary': 'rgba(255, 255, 255, 0.4)',
        'text-muted': 'rgba(255, 255, 255, 0.25)',

        // Semantic colors
        success: '#10B981',
        warning: '#F59E0B',
        danger: '#EF4444',
      },
      fontFamily: {
        sans: ['Inter', 'SF Pro Display', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['SF Pro Display', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        'display-xl': ['4.5rem', { lineHeight: '1', letterSpacing: '-0.03em', fontWeight: '700' }],
        'display-lg': ['3.5rem', { lineHeight: '1.1', letterSpacing: '-0.025em', fontWeight: '600' }],
        'display-md': ['2.5rem', { lineHeight: '1.2', letterSpacing: '-0.02em', fontWeight: '600' }],
      },
      backgroundImage: {
        // Premium gradients
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'gradient-mesh': `
          radial-gradient(at 40% 20%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
          radial-gradient(at 80% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
          radial-gradient(at 0% 50%, rgba(0, 212, 170, 0.1) 0px, transparent 50%),
          radial-gradient(at 80% 50%, rgba(168, 85, 247, 0.1) 0px, transparent 50%),
          radial-gradient(at 0% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 50%)
        `,
        'gradient-glow-primary': 'radial-gradient(ellipse at center, rgba(0, 212, 170, 0.15) 0%, transparent 70%)',
        'gradient-glow-accent': 'radial-gradient(ellipse at center, rgba(168, 85, 247, 0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        // Glow shadows
        'glow-sm': '0 0 20px rgba(0, 212, 170, 0.15)',
        'glow-md': '0 0 40px rgba(0, 212, 170, 0.2)',
        'glow-lg': '0 0 60px rgba(0, 212, 170, 0.25)',
        'glow-accent': '0 0 40px rgba(168, 85, 247, 0.2)',
        'glow-ai': '0 0 40px rgba(59, 130, 246, 0.2)',
        // Elevated shadows
        'elevated-sm': '0 4px 20px rgba(0, 0, 0, 0.3)',
        'elevated-md': '0 8px 40px rgba(0, 0, 0, 0.4)',
        'elevated-lg': '0 16px 60px rgba(0, 0, 0, 0.5)',
        // Glass shadows
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
        'glass-hover': '0 16px 48px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      backdropBlur: {
        xs: '2px',
        '2xl': '40px',
        '3xl': '64px',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'gradient-shift': 'gradient-shift 8s ease infinite',
        'fade-in': 'fade-in 0.5s ease-out',
        'slide-up': 'slide-up 0.5s ease-out',
        'scale-in': 'scale-in 0.3s ease-out',
        'spin-slow': 'spin 8s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'pulse-glow': {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.05)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        'gradient-shift': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      transitionDuration: {
        '400': '400ms',
      },
      transitionTimingFunction: {
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },
    },
  },
  plugins: [],
}
