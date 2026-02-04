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
        // CLINICAL FUTURISM - Ceramic Color System
        surface: '#FAFAFA',        // Ceramic White - Base background
        'surface-highlight': '#FFFFFF',  // Pure white for elevated cards
        
        // Text Hierarchy
        'text-primary': '#121212',    // Soft high-contrast black
        'text-secondary': '#686868',  // Graphite gray for labels
        'text-mono': '#454545',       // For data/numbers
        
        // Functional Accents
        'accent-medical': '#0F766E',  // Surgical Teal - Medical features
        'accent-ai': '#2563EB',       // Electric Blue - AI/Scanning
        
        // Border System
        'border-subtle': 'rgba(0,0,0,0.04)',  // Ultra-faint borders
        'border-glass': 'rgba(255,255,255,0.2)', // Glass highlights
        
        // Extended Palette for Compatibility
        medical: {
          50: '#f0fdfa',
          100: '#ccfbf1',
          500: '#0F766E',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        ai: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#2563EB',
          600: '#3b82f6',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      letterSpacing: {
        tighter: '-0.025em',  // Technical feel
        tight: '-0.015em',
      },
      boxShadow: {
        // Ambient Occlusion Shadows
        'glass': '0 0 0 1px rgba(0,0,0,0.03), 0 2px 8px rgba(0,0,0,0.04)',
        'float': '0 12px 24px -8px rgba(0,0,0,0.08), 0 4px 8px -4px rgba(0,0,0,0.04)',
        'dock': '0 8px 32px -4px rgba(0,0,0,0.12), 0 0 0 1px rgba(255,255,255,0.1)',
      },
      backdropBlur: {
        'xs': '2px',
        '2xl': '24px',
      },
      borderRadius: {
        'ceramic': '24px',  // Smooth organic curves
      },
    },
  },
  plugins: [],
}
