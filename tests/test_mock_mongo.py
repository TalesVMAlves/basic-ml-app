import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
from bson.objectid import ObjectId
from app.app import app, conditional_auth
from db.engine import get_mongo_collection 
from intent_classifier import IntentClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="function")
def mock_auth(monkeypatch):    
    def override_auth():
        """Usuário de teste mockado"""
        return "test_user_mocked"
    
    app.dependency_overrides[conditional_auth] = override_auth
    
    yield

    app.dependency_overrides = {}

@pytest.fixture(scope="function")
def mock_models(monkeypatch):
    mock_model = MagicMock(spec=IntentClassifier)
    
    mock_model.predict.return_value = ("mock_intent", {"mock_intent": 0.99, "other_intent": 0.01})
    
    mock_models_dict = {"mock_model_v1": mock_model}
    
    monkeypatch.setattr("app.app.MODELS", mock_models_dict)
    
    return mock_models_dict, mock_model

@pytest.fixture(scope="function")
def mock_db_collection(monkeypatch):
    """
    Mocks o objeto 'collection' global em app.app.
    """
    mock_collection = MagicMock()

    def insert_one_side_effect(doc_to_insert):
        """Simula o comportamento do pymongo.insert_one"""
        inserted_id = ObjectId("691283dfa2a42c9e7c7529d1")
        doc_to_insert['_id'] = inserted_id
        return MagicMock(inserted_id=inserted_id)

    mock_collection.insert_one.side_effect = insert_one_side_effect

    monkeypatch.setattr("app.app.collection", mock_collection)

    return mock_collection


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "is running" in data["message"]

def test_post_predict_unit(client, mock_auth, mock_models, mock_db_collection):
    """
    Testa a rota POST /predict (Teste de Unidade).
    Usa mocks para autenticação, modelos e banco de dados.
    Verifica se os mocks foram chamados corretamente.
    """
    mock_models_dict, mock_model_instance = mock_models
    
    test_text = "oi tudo bem"
    response = client.post("/predict", params={"text": test_text})
    
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == test_text
    assert data["owner"] == "test_user_mocked" 
    assert data["id"] == "691283dfa2a42c9e7c7529d1" 
    
    assert "mock_model_v1" in data["predictions"]
    assert data["predictions"]["mock_model_v1"]["top_intent"] == "mock_intent"
    
    mock_model_instance.predict.assert_called_once_with(test_text)
    
    mock_db_collection.insert_one.assert_called_once()
    insert_args = mock_db_collection.insert_one.call_args[0][0] 
    assert insert_args["text"] == test_text
    assert insert_args["owner"] == "test_user_mocked"
    assert "timestamp" in insert_args
    assert insert_args["predictions"]["mock_model_v1"]["top_intent"] == "mock_intent"
    assert "_id" in insert_args 


@pytest.mark.integration
def test_post_predict_integration_db(client, mock_auth, mock_models):

    env = os.getenv("ENV", "prod").lower()
    collection_name = f"{env.upper()}_intent_logs"
    real_collection = get_mongo_collection(collection_name)
    assert real_collection is not None, f"Falha ao conectar no DB de integração (Coleção: {collection_name})"

    mock_models_dict, mock_model_instance = mock_models
    
    test_text = f"integration_test_{os.urandom(4).hex()}" 
    response = client.post("/predict", params={"text": test_text})
    
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == test_text
    assert "id" in data
    inserted_id_str = data["id"]
    
    doc_id = None
    try:
        doc_id = ObjectId(inserted_id_str)
        doc = real_collection.find_one({"_id": doc_id})
        
        assert doc is not None, "Documento não encontrado no DB"
        assert doc["text"] == test_text
        assert doc["owner"] == "test_user_mocked" 
        assert doc["predictions"]["mock_model_v1"]["top_intent"] == "mock_intent"
    
    finally:
        if doc_id:
            real_collection.delete_one({"_id": doc_id})
            print(f"\nLimpeza: Deletado documento de teste {doc_id}")
            
            doc_after = real_collection.find_one({"_id": doc_id})
            assert doc_after is None, "Falha ao limpar o documento de teste"