from routers.health import router as health_router
from routers.ask import router as ask_router
from fastapi import FastAPI

app = FastAPI(title="OKUULIB")

app.include_router(health_router)
app.include_router(ask_router)

