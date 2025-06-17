import os
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

USE_FIREBASE = os.getenv("USE_FIREBASE", "true").lower() == "true"

# Only create HTTPBearer if using Firebase
bearer_scheme = HTTPBearer() if USE_FIREBASE else None

if USE_FIREBASE:
    from pathlib import Path
    import firebase_admin
    from firebase_admin import credentials, auth

    # Initialize Firebase Admin only once
    cred = credentials.Certificate(Path(__file__).parent / "serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

    def _verify_firebase_user(
        http_credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> str:
        token = http_credentials.credentials
        try:
            decoded_token = auth.verify_id_token(token)
            return str(decoded_token["uid"])
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid authentication") from e

else:

    def _verify_demo_user(request: Request) -> str:
        demo_user = request.headers.get("X-Demo-User")
        if not demo_user:
            raise HTTPException(status_code=401, detail="Missing X-Demo-User header")
        return demo_user


def verify_user(
    request: Request,
    http_credentials: HTTPAuthorizationCredentials | None = (
        Depends(bearer_scheme) if USE_FIREBASE else None
    ),
) -> str:
    if USE_FIREBASE:
        if http_credentials is not None:
            return _verify_firebase_user(http_credentials)
        else:
            raise HTTPException(status_code=401, detail="Invalid authentication")
    else:
        return _verify_demo_user(request)
