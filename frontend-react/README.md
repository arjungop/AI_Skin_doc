# AI Skin Doctor – React Frontend

A modern, React + Vite frontend for the AI Skin Doctor project.

## Quick start

1) Install Node.js 18+
2) Copy env and set API base URL

```
cp .env.example .env
# edit .env to point to your backend
# VITE_API_BASE_URL=http://127.0.0.1:8000
```

3) Install deps and run

```
npm install
npm run dev
# or build & preview
npm run build
npm run preview
```

## Pages
- Login, Register
- Patient Dashboard (cards to Lesions, AI Chat, Appointments, Transactions)
- Doctor Dashboard
- Lesion Upload & Prediction (with AI diagnosis)
- AI Chat (LLM)

## Notes
- Auth/session stored in localStorage: `patient_id`, `user_id`, `username`, `role`.
- API base URL taken from `import.meta.env.VITE_API_BASE_URL`.
- Works with the FastAPI backend in `backend/`.

