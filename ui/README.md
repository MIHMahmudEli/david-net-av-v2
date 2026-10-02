# DAVID-Net Web UI (Next.js)

End-user interface: upload a video → see per-modality (video / audio) verdicts, the
authenticity quadrant, temporal manipulation timelines, and the AV-sync curve.

## Stack
- Next.js (App Router) + TypeScript
- Tailwind CSS
- A charting lib for the dual timeline + sync curve (e.g. Recharts or lightweight SVG)
- Calls the FastAPI HF Space via a **server-side route handler** (keeps the API URL/token off the client and sidesteps CORS)

## Development Setup

```bash
cd ui
npm install
# Ensure .env.local has the FastAPI backend endpoint:
# API_URL=http://localhost:7860
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Production Build

```bash
npm run build
npm start
```

## Production Docker Deployment

```bash
docker build -t davidnet-ui:latest -f ui/Dockerfile ui/
docker run -p 3000:3000 -e API_URL=http://your-fastapi-host:7860 davidnet-ui:latest
```

Or run the full stack (API + UI) using the root docker-compose:
```bash
docker compose up --build
```

## Structure
```
ui/
  app/
    globals.css              # Dark forensic Tailwind styles
    layout.tsx               # Root layout & metadata
    page.tsx                 # Full forensic workbench (Uploader, Verdicts, Timelines, Sync Curve)
    api/predict/route.ts     # Edge proxy forwarding video to FastAPI /predict
    api/predict-audio/route.ts # Edge proxy forwarding audio to FastAPI /predict-audio
  components/
    Uploader.tsx             # Drag-drop file uploader with size/duration validation
    VerdictCard.tsx          # Modality cards with circular calibrated gauges
    QuadrantBadge.tsx        # 4-quadrant attribution badges (RVRA, RVFA, FVRA, FVFA)
    DualTimeline.tsx         # Multi-track forensic timeline showing manipulated intervals
    SyncCurve.tsx            # SVG cross-modal synchrony curve visualization
    Disclaimer.tsx           # Responsible AI disclosure & threshold info
  lib/api.ts                 # Strongly typed TypeScript API client
```

## Deploy to Vercel
1. Push repository to GitHub.
2. Import the `ui/` directory in Vercel.
3. Configure Environment Variable: `API_URL` pointing to your Hugging Face Space (e.g. `https://<space-name>.hf.space`).
4. Click **Deploy**.
