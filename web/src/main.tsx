import './polyfills.ts';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App.tsx';
import { Splash } from './components/Splash.tsx';

// visitor counts (Vercel Web Analytics: no cookies, no personal data). Vercel
// serves the script only once Analytics is switched on for the project, so it
// loads only in builds made with VITE_ANALYTICS=1 (set in the project's
// environment variables alongside switching it on), and only on the live site
if (import.meta.env.VITE_ANALYTICS === '1' && location.hostname === 'ag-glass-box.vercel.app') {
  const s = document.createElement('script');
  s.defer = true;
  s.src = '/_vercel/insights/script.js';
  document.head.appendChild(s);
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Splash />
    <App />
  </StrictMode>,
);
