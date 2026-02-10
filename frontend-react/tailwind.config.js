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
        // MODERN CONSUMER HEALTH PALETTE - Soft, Trustworthy, Premium

        // Backgrounds
        background: '#F8FAFC',     // Clean slate-50
        surface: '#FFFFFF',        // Pure white
        'surface-subtle': '#F1F5F9', // Slate-100 for sidebars/panels

        // Primary Brand - Soft Coral (Warmth & Health)
        primary: {
          50: '#FFF1F2',
          100: '#FFE4E6',
          200: '#FECDD3',
          300: '#FDA4AF',
          400: '#FB7185',
          500: '#F43F5E', // Rose-500: Warm, human, medical but friendly
          600: '#E11D48',
          700: '#BE123C',
          800: '#9F1239',
          900: '#881337',
        },

        // Secondary - Calming Teal/Slate (Trust)
        secondary: {
          50: '#F0F9FF',
          100: '#E0F2FE',
          200: '#BAE6FD',
          300: '#7DD3FC',
          400: '#38BDF8',
          500: '#0EA5E9', // Sky-500
          600: '#0284C7',
          700: '#0369A1',
          800: '#075985',
          900: '#0C4A6E',
        },

        // Text - High Contrast but Soft
        text: {
          primary: '#0F172A',   // Slate-900 (Not pure black)
          secondary: '#475569', // Slate-600
          tertiary: '#94A3B8',  // Slate-400
          muted: '#CBD5E1',     // Slate-300
        },

        // Semantic
        success: '#10B981', // Emerald
        warning: '#F59E0B', // Amber
        danger: '#EF4444',  // Red
        info: '#3B82F6',    // Blue
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'], // Clean, standard sans
        display: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'], // Modern geometric for headings
      },
      boxShadow: {
        'soft-sm': '0 2px 4px rgba(148, 163, 184, 0.05)',
        'soft-md': '0 4px 6px -1px rgba(148, 163, 184, 0.1), 0 2px 4px -1px rgba(148, 163, 184, 0.06)',
        'soft-lg': '0 10px 15px -3px rgba(148, 163, 184, 0.1), 0 4px 6px -2px rgba(148, 163, 184, 0.05)',
        'soft-xl': '0 20px 25px -5px rgba(148, 163, 184, 0.1), 0 10px 10px -5px rgba(148, 163, 184, 0.04)',
        'glow': '0 0 20px rgba(244, 63, 94, 0.15)', // Subtle primary glow
      },
      borderRadius: {
        '3xl': '1.5rem',
        '4xl': '2rem',
        'pill': '9999px',
      },
      backgroundImage: {
        'gradient-soft': 'linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)',
        'gradient-primary': 'linear-gradient(135deg, #F43F5E 0%, #E11D48 100%)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
