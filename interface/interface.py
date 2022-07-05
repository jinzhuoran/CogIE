import torch
from cogie import *
from cogie.utils.util import load_yaml
from flask import Flask, jsonify, request
from flask_cors import *

tokenize_toolkit = TokenizeToolkit()

ner_toolkit = NerToolkit(corpus="conll2003")

el_toolkit = ElToolkit(corpus="wiki")

et_toolkit = EtToolkit(corpus="ufet")

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["JSON_AS_ASCII"] = False


@app.route("/ner", methods=["GET", "POST"])
def ner():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    return jsonify({"sentence": " ".join(words), "ner_result": ner_result})


@app.route("/linking", methods=["GET", "POST"])
def linking():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    el_result = el_toolkit.run(ner_result)
    return jsonify({
        "sentence": " ".join(words),
        "el_result": [{"mention":entity["mention"],"text":entity["text"]} for entity in el_result]}
    )


@app.route("/typing", methods=["GET", "POST"])
def typing():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    et_result = et_toolkit.run(ner_result)
    return jsonify({"sentence": " ".join(words), "et_result": et_result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9988)
