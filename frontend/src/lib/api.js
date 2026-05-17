// API base URL — set VITE_API_URL in .env.local for local dev
// Copy .env.example → .env.local and adjust if needed.
export const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
