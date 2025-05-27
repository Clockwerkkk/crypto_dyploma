from fastapi import FastAPI
from app.api import router

app = FastAPI(title="Crypto MCP Server")

app.include_router(router)
