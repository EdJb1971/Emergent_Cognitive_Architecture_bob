const directBackendUrl = import.meta.env.VITE_BACKEND_URL;

export const API_BASE_URL = directBackendUrl || '';
export const API_KEY = import.meta.env.VITE_API_KEY || (!directBackendUrl ? 'local-vite-proxy' : undefined);
export const WS_BASE_URL = directBackendUrl
  ? directBackendUrl.replace(/^http/, 'ws')
  : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
