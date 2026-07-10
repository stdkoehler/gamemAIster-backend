"""Run FastAPI Server"""

import os
import uvicorn

ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() == "true"

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        workers=2,
        ssl_keyfile="cert/key.pem" if ssl_enabled else None,
        ssl_certfile="cert/cert.pem" if ssl_enabled else None,
    )
