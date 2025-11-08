import os
import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.app import app

@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)

@pytest.fixture(scope="module")
def mongo_test_client():
    try:
        mongo_uri = os.getenv("MONGO_URI_TEST", "mongodb://localhost:27017/")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("\nConexão com MongoDB de teste bem-sucedida.")
        db = client["test_db"]
        yield db
        print("\nLimpando banco de dados de teste...")
        client.drop_database("test_db")
        client.close()
    except Exception as e:
        pytest.skip(f"Não foi possível conectar ao MongoDB para testes de integração: {e}")


@patch('app.app.IntentClassifier')
@patch('app.app.get_mongo_collection')
def test_predict_endpoint_success_mocked(mock_get_collection, mock_intent_classifier, test_client):
    """
    Testa o endpoint /predict com sucesso, mockando o modelo e o banco de dados.
    """
    mock_classifier_instance = MagicMock()
    mock_classifier_instance.predict.return_value = ("confusion", {"confusion": 0.9, "certainty": 0.1})
    mock_intent_classifier.return_value = mock_classifier_instance

    mock_collection = MagicMock()
    mock_get_collection.return_value = mock_collection

    response = test_client.post("/predict?text=Isso é um teste")

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Isso é um teste"
    assert data["owner"] == "dev_user"
    assert "confusion-v1" in data["predictions"]
    assert data["predictions"]["confusion-v1"]["top_intent"] == "confusion"
    
    mock_collection.insert_one.assert_called_once()


def test_root_endpoint(test_client):
    """
    Testa o endpoint raiz (/)
    """
    response = test_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()



@patch('app.app.IntentClassifier')
def test_predict_endpoint_integration_with_db(mock_intent_classifier, test_client, mongo_test_client):
    """
    Testa a integração do endpoint /predict com um banco de dados MongoDB real.
    """
    mock_classifier_instance = MagicMock()
    mock_classifier_instance.predict.return_value = ("certainty", {"certainty": 0.98, "confusion": 0.02})
    mock_intent_classifier.return_value = mock_classifier_instance
    
    with patch('app.app.collection', mongo_test_client["test_logs"]):
        response = test_client.post("/predict?text=Teste de integração com DB")

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Teste de integração com DB"

        log_entry = mongo_test_client["test_logs"].find_one({"text": "Teste de integração com DB"})
        assert log_entry is not None
        assert log_entry["owner"] == "dev_user"
        assert log_entry["predictions"]["confusion-v1"]["top_intent"] == "certainty"