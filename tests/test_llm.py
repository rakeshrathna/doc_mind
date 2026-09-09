import pytest
from unittest.mock import patch, MagicMock
import requests
from llm import parse_json, check_ollama_status, chat
from exceptions import LLMParseError, OllamaUnavailableError, ModelNotFoundError


def test_json_parsing_pure():
    raw = '{"vendor": "ACME", "total": 100.0}'
    data = parse_json(raw)
    assert data["vendor"] == "ACME"
    assert data["total"] == 100.0


def test_json_parsing_markdown():
    raw = 'Here is the JSON:\n```json\n{"vendor": "ACME", "total": 100.0}\n```'
    data = parse_json(raw)
    assert data["vendor"] == "ACME"
    assert data["total"] == 100.0


def test_json_parsing_surrounding_text():
    raw = 'Note: The extracted vendor is ACME. {"vendor": "ACME", "total": 100.0} Hope this helps!'
    data = parse_json(raw)
    assert data["vendor"] == "ACME"
    assert data["total"] == 100.0


def test_json_parsing_malformed():
    raw = 'No json content present in this string.'
    with pytest.raises(LLMParseError):
        parse_json(raw)


@patch("requests.get")
def test_ollama_unavailable(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    with pytest.raises(OllamaUnavailableError):
        check_ollama_status()


@patch("requests.get")
def test_model_not_found(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "mistral:latest"}]}
    mock_get.return_value = mock_resp
    
    with pytest.raises(ModelNotFoundError):
        check_ollama_status(model="llama3.1:8b")
