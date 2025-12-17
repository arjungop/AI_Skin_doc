/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Legacy dark tokens (still used by older pages)
        bg: '#0f172a',
        panel: '#111827',
        text: '#e5e7eb',
        muted: '#9ca3af',
        accent2: '#7c3aed',
        border: '#1f2937',

        // New Azure-themed palette
        primary: '#0078D4',      // Azure Blue
        accent: '#6B5BFF',       // Accent Purple
        coral: '#FF6B6B',        // Alert accent
        secondary: '#00B2A9',    // Teal
        teal: '#00B2A9',
        background: '#F9FAFB',   // Light background
        backgroundDark: '#1F2937',
        card: '#FFFFFF',
        textDark: '#1E293B',     // Slate 800
        textMuted: '#64748B',    // Slate 500
        textPrimary: '#1E293B',
        textSecondary: '#64748B',
        coolGray: '#6B7280',     // Neutral label
        warmGray: '#D1D5DB',     // Neutral border/muted
        borderGray: '#E2E8F0',
        borderLight: '#E5E7EB',
        error: '#DC2626',
        warning: '#F59E0B',
        success: '#10B981',
        azure100: '#E0F2FE',
        // Luxury palette
        backgroundSoft: '#F3F2F7',
        backgroundSecondary: '#E8E6EF',
        cardBackground: '#F7F6FB',
        accentGold: '#C6A664',
        primaryBlue: '#4E73DF',
        accentPurple2: '#6A5BFF',
        textLuxury: '#2C2E33',
        textLuxuryMuted: '#5F6368',
        borderElegant: '#D5D8DE',
        successLux: '#2FAE7B',
        warningLux: '#E9A44F',
        errorLux: '#D64550',
      },
    },
  },
  plugins: [],
}
