import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './smoothScroll.js' // Added lenis smooth scrolling
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
