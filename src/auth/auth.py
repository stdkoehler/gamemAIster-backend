import os
from pathlib import Path

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import firebase_admin
from firebase_admin import credentials, auth

USE_FIREBASE = os.getenv("USE_FIREBASE", "true").lower() == "true"

# Only create HTTPBearer if using Firebase
bearer_scheme = HTTPBearer() if USE_FIREBASE else None

if USE_FIREBASE:
    cred = credentials.Certificate(Path(__file__).parent / "serviceAccountKey.json")
    firebase_admin.initialize_app(cred)


async def firebase_user(
    http_credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> str:
    try:
        decoded_token = auth.verify_id_token(http_credentials.credentials)
        return str(decoded_token["uid"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")


async def demo_user(request: Request) -> str:
    user = request.headers.get("X-Demo-User")
    if not user:
        raise HTTPException(status_code=401, detail="Missing X-Demo-User header")
    return user


async def verify_user(
    request: Request,
    http_credentials: HTTPAuthorizationCredentials | None = (
        Depends(HTTPBearer()) if USE_FIREBASE else Depends(lambda: None)
    ),
) -> str:
    if USE_FIREBASE:
        if not http_credentials:
            raise HTTPException(status_code=401, detail="Missing auth token")
        return await firebase_user(http_credentials)
    else:
        return await demo_user(request)
