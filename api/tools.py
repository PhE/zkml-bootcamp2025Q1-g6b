"""
Usefull tools
"""

import io
import os
import sys
import zipfile

import psutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

origins = [
    # TODO: add github pages FQDN
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8889",
    "https://zkg6-front.s3-website.nl-ams.scw.cloud",
    "http://zkg6-front.s3-website.nl-ams.scw.cloud",
]


def cors_security(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def stream_output(f):
    """Capture stdout/stderr and stream it"""
    buffer = io.StringIO()
    sys.stdout = buffer
    sys.stderr = buffer
    try:
        # Run the long-running task
        await f()
    finally:
        # Restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # Yield the captured output
    buffer.seek(0)
    for line in buffer:
        yield line


def stream_response(f):
    """return the streamed output as fastapi response"""
    return StreamingResponse(stream_output(f), media_type="text/plain")


def server_info():
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")

    return {
        "cpu": {
            "count": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
        },
        "memory": {
            "total_ram": memory_info.total,
            "available_ram": memory_info.available,
            "used_ram": memory_info.used,
            "ram_percent": memory_info.percent,
        },
        "disk": {
            "total_disk": disk_info.total,
            "used_disk": disk_info.used,
            "disk_percent": disk_info.percent,
        },
        "uptime": psutil.boot_time(),
    }


def zipfile_stream(
    zip_name: str,
    zip_subdir: str,
    filenames: list[str],
):
    zip_io = io.BytesIO()
    with zipfile.ZipFile(
        zip_io, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as temp_zip:
        for fpath in filenames:
            # Calculate path for file in zip
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            temp_zip.write(fpath, zip_path)
    return StreamingResponse(
        iter([zip_io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={zip_name}.zip"},
    )
