"""
@Author: jinzhuan
@File: interface.py
@Desc: 
"""
from cogie import *
from flask import Flask, jsonify, request
from flask_cors import *

tokenize_toolkit = TokenizeToolkit(task='ws', language='english', corpus=None)

ner_toolkit = NerToolkit(task='ner', language='english', corpus='trex')

ee_ner_toolkit = NerToolkit(task='ner', language='english', corpus='ace2005')

et_toolkit = EtToolkit(task='et', language='english', corpus=None)

el_toolkit = ElToolkit()

re_toolkit = ReToolkit(task='re', language='english', corpus='trex')

ee_toolkit = EeToolkit(task='ee', language='english', corpus='ace2005')

fn_toolkit = FnToolkit(task='fn', language='english', corpus=None)

argument_toolkit = ArgumentToolkit(task='fn', language='english', corpus='argument')

app = Flask(__name__, static_url_path='')
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False


@app.route('/ner', methods=["GET", "POST"])
def ner():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    return jsonify({"words": words, "ner_result": ner_result})


@app.route('/typing', methods=["GET", "POST"])
def typing():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    et_result = et_toolkit.run(words, ner_result)
    return jsonify({"words": words, "ner_result": ner_result, "et_result": et_result})


@app.route('/linking', methods=["GET", "POST"])
def linking():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    el_result = el_toolkit.run(sentence)
    return jsonify({"words": words, "el_result": el_result})


@app.route('/relation', methods=["GET", "POST"])
def relation():
    sentence = request.values.get('sentence')
    words = tokenize_toolkit.run(sentence)
    ner_result = ner_toolkit.run(words)
    re_result = re_toolkit.run(words, ner_result)
    return jsonify({"words": words, "ner_result": ner_result, "re_result": re_result})


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
    return jsonify({"words": words, "ner_result": ner_result, "ee_result": ee_result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9988)
