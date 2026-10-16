import './polyfills.ts';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@xyflow/react/dist/style.css';
import './styles.css';
import { App } from './App.tsx';
import { Splash } from './components/Splash.tsx';

// visitor counts (Vercel Web Analytics: no cookies, no personal data). Only on
// the live site, so local builds and tests send nothing; the script is served
// by Vercel once analytics is switched on for the project
if (location.hostname === 'ag-glass-box.vercel.app') {
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
