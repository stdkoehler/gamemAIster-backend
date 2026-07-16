"""Encryption at rest for user-supplied secrets (LLM API keys)."""

import os
from functools import lru_cache

from cryptography.fernet import Fernet


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = os.getenv("SETTINGS_ENCRYPTION_KEY")
    if not key:
        raise ValueError(
            "SETTINGS_ENCRYPTION_KEY is not set — required to store user LLM API keys"
        )
    return Fernet(key.encode())


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()
