import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api проксируется на FastAPI-бэкенд (по умолчанию localhost:8000)
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: { '/api': process.env.BACKEND_URL || 'http://localhost:8000' },
  },
})
