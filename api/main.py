"""
The zkg6 API service

The API endpoints are defined here.
"""

from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import tools
from .model import fashion

app = FastAPI(
    title="zkml group 6 api",
    description="API for the ZKML app from group 6",
    # openapi_prefix=api_prefix,
)
tools.cors_security(app)

app.mount("/front", StaticFiles(directory=".output/public"), name="static")


# ---------------------------------------------------------------------------------------


@app.get("/api/fashion/setup", tags=["fashion"])
async def get_model_setup():
    return await fashion.setup()


@app.post("/api/fashion/prover", tags=["fashion"])
async def get_prover_step(
    input_img: UploadFile,
):
    return await fashion.prover_step(input_img)


@app.post("/api/fashion/verifier", tags=["fashion"])
async def get_verifier_step(
    proof: UploadFile,
):
    return await fashion.verifier_step(proof)


@app.get("/api/fashion/image/{image_idx}", tags=["fashion"])
def get_image(image_idx: int):
    return fashion.get_image(image_idx)


# ---------------------------------------------------------------------------------------

# root redirects to docs
@app.get("/", tags=["misc"])
def home():
    return RedirectResponse("/front/en.html")


# usual health check endpoint
@app.get(
    "/api/status",
    description="Get server status/info",
    tags=["misc"],
)
def status():
    return tools.server_info()


# ---------------------------------------------------------------------------------------
