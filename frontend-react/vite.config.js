import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    open: true,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-ui': ['framer-motion', 'react-icons', 'lucide-react', 'clsx', 'tailwind-merge'],
          'vendor-3d': ['three', '@react-three/fiber', '@react-three/drei'],
          'vendor-map': ['leaflet', 'react-leaflet'],
          'vendor-markdown': ['react-markdown'],
        },
      },
    },
  },
})

