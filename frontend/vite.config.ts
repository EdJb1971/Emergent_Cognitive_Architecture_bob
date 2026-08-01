import { fileURLToPath, URL } from 'node:url';
import { defineConfig, loadEnv, ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
  const localEnvironment = loadEnv(mode, repositoryRoot, '');
  const backendTarget = localEnvironment.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
  const proxyHeaders = localEnvironment.API_KEY
    ? { 'X-API-Key': localEnvironment.API_KEY }
    : undefined;
  const apiProxy: ProxyOptions = {
    target: backendTarget,
    changeOrigin: true,
    headers: proxyHeaders,
  };

  return {
    plugins: [react()],
    resolve: {
      alias: {
        api: fileURLToPath(new URL('./src/api', import.meta.url)),
        components: fileURLToPath(new URL('./src/components', import.meta.url)),
        types: fileURLToPath(new URL('./src/types', import.meta.url)),
      },
    },
    server: {
      host: '127.0.0.1',
      port: 3000,
      strictPort: true,
      proxy: {
        '/api': apiProxy,
        '/chat': apiProxy,
        '/ws': { ...apiProxy, ws: true },
      },
    },
    preview: {
      host: '127.0.0.1',
      port: 4173,
      strictPort: true,
      proxy: {
        '/api': apiProxy,
        '/chat': apiProxy,
        '/ws': { ...apiProxy, ws: true },
      },
    },
    build: {
      outDir: 'build',
      sourcemap: false,
    },
  };
});
