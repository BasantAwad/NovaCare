from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import medication, telemetry

app = FastAPI(title="NovaCare Unified API")

# Allow CORS for frontend and mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(medication.router)
app.include_router(telemetry.router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "NovaCare API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
