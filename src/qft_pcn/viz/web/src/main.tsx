import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
// KaTeX CSS is imported globally so Task 4 panels can render math.
import 'katex/dist/katex.min.css';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
