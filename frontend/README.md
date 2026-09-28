# MISSIONCTRL command center

Start the FastAPI service on port 8000, then run:

```powershell
npm.cmd install
npm.cmd run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`, so the client uses
the existing mission API without requiring browser CORS configuration.
