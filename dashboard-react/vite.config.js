import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxy = {
  '/api': {
    target: 'http://localhost:5001',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { ...apiProxy },
    // Allow importing project-root `changed.md` as raw fallback for Architecture panel
    fs: { allow: ['..'] },
  },
  // `vite preview` does not inherit `server.proxy`; without this, /api/* returns 404
  // unless the app is served from Flask (port 5001) with the API.
  preview: {
    port: 4173,
    proxy: { ...apiProxy },
  },
})
