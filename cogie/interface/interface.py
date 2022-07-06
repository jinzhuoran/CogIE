import torch
from cogie import *
from cogie.utils.util import load_yaml
from flask import Flask, jsonify, request
from flask_cors import *

tokenize_toolkit = TokenizeToolkit()

ner_toolkit = NerToolkit(corpus="conll2003")
el_toolkit = ElToolkit(corpus="wiki")
et_toolkit = EtToolkit(corpus="ufet")
re_ner_toolkit = NerToolkit(corpus="trex")
re_toolkit = ReToolkit(task='re', language='english', corpus='trex')
ee_ner_toolkit = NerToolkit(corpus="ace2005")
ee_toolkit = EeToolkit(task='ee', language='english', corpus='ace2005')
fn_toolkit = FnToolkit(task='fn', language='english', corpus='frame')
argument_toolkit = ArgumentToolkit(task='fn', language='english', corpus='argument')

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["JSON_AS_ASCII"] = False


@app.route("/ner", methods=["GET", "POST"])
def ner():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    return jsonify({
        "words": words,
        "ner_result": [
            {
                "mention":words[entity["start"]:entity["end"]],
                "start":entity["start"],
                "end":entity["end"],
            }
            for entity in ner_result
        ]
    })


@app.route("/linking", methods=["GET", "POST"])
def linking():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    el_result = el_toolkit.run(ner_result)
    return jsonify({
        "words": words,
        "el_result":  [
            {
                "mention":words[entity["start"]:entity["end"]],
                "start":entity["start"],
                "end":entity["end"],
                "title":entity["title"],
                "text": entity["text"],
                "url": entity["url"],
                "cognet_link": entity["cognet_link"],
            }
            for entity in el_result
        ]
    })

@app.route("/typing", methods=["GET", "POST"])
def typing():
    sentence = request.values.get("sentence")
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    et_result = et_toolkit.run(ner_result)
    return jsonify({
        "words": words,
        "et_result":  [
            {
                "mention":words[entity["start"]:entity["end"]],
                "start":entity["start"],
                "end":entity["end"],
                "type":entity["types"]
            }
            for entity in et_result
        ]
    })


@app.route('/relation', methods=["GET", "POST"])
def relation():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    ner_result = re_ner_toolkit.run(words)
    re_result = re_toolkit.run(words, ner_result)
    return jsonify({
        "words": words,
        "ner_result": [
            {
                "mention":words[entity["start"]:entity["end"]],
                "start":entity["start"],
                "end":entity["end"],
            }
            for entity in ner_result
        ],
        "re_result": re_result
    })


@app.route('/frame', methods=["GET", "POST"])
def frame():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    fn_result = fn_toolkit.run(words)
    element_result = argument_toolkit.run(words, fn_result)
    return jsonify({"words": words, "fn_result": fn_result, 'element_result': element_result})


@app.route('/event', methods=["GET", "POST"])
def event():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    ner_result = ee_ner_toolkit.run(words)
    ee_result = ee_toolkit.run(words, ner_result)
    return jsonify({"words": words, "ee_result": ee_result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9988)
