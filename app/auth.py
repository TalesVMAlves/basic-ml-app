"""

# Criar um novo token
python app/auth.py create --owner="alguem" --expires_in_days=365

# Ler todos os tokens
python app/auth.py read_all

"""

import uuid
from datetime import datetime, timedelta
from db.engine import get_mongo_collection
from fastapi import Request, HTTPException


class TokenManager:
    """
    Gerencia tokens da API.
    """
    def create(self, owner: str, note: str = "", expires_in_days: int = 180):
        """
        Cria um novo token com tempo de expiração.

        Args:
            owner (str): Nome do dono do token.
            note (str): Descrição.
            expires_in_days (int): Validade do token em dias.
        """
        token = str(uuid.uuid4())
        tokens_collection = get_mongo_collection("api_tokens")

        now = datetime.utcnow()
        token_doc = {
            "token": token,
            "owner": owner,
            "note": note,
            "created_at": now,
            "expires_at": now + timedelta(days=expires_in_days),
            "active": True
        }

        tokens_collection.insert_one(token_doc)
        print(f"✅ Token criado (expira em {expires_in_days} dias): {token}")

    def read_all(self):
        """
        Lê e imprime todos os tokens armazenados no MongoDB.
        """
        tokens_collection = get_mongo_collection("api_tokens")
        all_tokens = tokens_collection.find()
        for t in all_tokens:
            print({
                "token": t.get("token"),
                "owner": t.get("owner"),
                "note": t.get("note"),
                "active": t.get("active"),
                "created_at": t.get("created_at")
            })

    def delete_expired(self):
        """
        Remove tokens expirados da base.
        """
        tokens_collection = get_mongo_collection("api_tokens")
        result = tokens_collection.delete_many({"expires_at": {"$lt": datetime.utcnow()}})
        print(f"🧹 Tokens expirados removidos: {result.deleted_count}")



def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = token.replace("Bearer ", "")
    tokens_collection = get_mongo_collection("api_tokens")
    token_entry = tokens_collection.find_one({"token": token, "active": True})

    if not token_entry:
        raise HTTPException(status_code=403, detail="Invalid or inactive token")

    if datetime.utcnow() > token_entry["expires_at"]:
        raise HTTPException(status_code=403, detail="Token expired")

    return token_entry["owner"]


if __name__ == "__main__":
    import fire
    fire.Fire(TokenManager)