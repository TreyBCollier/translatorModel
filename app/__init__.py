from train import translate
from trainEnglishToFrench import translateEnglishToFrench

import flask


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def index():
    return "Loading translation..."


@app.route('/translateFrench/<sentence>')
def translateFrench(sentence):
    return translate(sentence)


@app.route('/translateEnglish/<sentence>')
def translateEnglish(sentence):
    return translateEnglishToFrench(sentence)
