import json
import sys
import os
from unittest.mock import patch

# Legg til lambda-mappen i path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lambda")))

from app import handler


@patch("app.boto3.client")
def test_positive_sentiment(mock_boto):

    mock_client = mock_boto.return_value
    mock_client.detect_sentiment.return_value = {
        "Sentiment": "POSITIVE"
    }

    event = {"body": "I love this product"}

    response = handler(event, None)

    assert response["statusCode"] == 200

    body = json.loads(response["body"])
    sentiment_json = json.loads(body["sentiment "])

    assert sentiment_json["Sentiment"] == "POSITIVE"


@patch("app.boto3.client")
def test_negative_sentiment(mock_boto):

    mock_client = mock_boto.return_value
    mock_client.detect_sentiment.return_value = {
        "Sentiment": "NEGATIVE"
    }

    event = {"body": "This is terrible"}

    response = handler(event, None)

    body = json.loads(response["body"])
    sentiment_json = json.loads(body["sentiment "])

    assert sentiment_json["Sentiment"] == "NEGATIVE"


@patch("app.boto3.client")
def test_detect_sentiment_called_with_correct_parameters(mock_boto):

    mock_client = mock_boto.return_value
    mock_client.detect_sentiment.return_value = {
        "Sentiment": "NEUTRAL"
    }

    event = {"body": "It is okay"}

    handler(event, None)

    mock_client.detect_sentiment.assert_called_once_with(
        LanguageCode="en",
        Text="It is okay"
    )
