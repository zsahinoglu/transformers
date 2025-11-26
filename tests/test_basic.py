from transformers import greet


def test_greet():
    assert greet("Zafer") == "Hello, Zafer!"
