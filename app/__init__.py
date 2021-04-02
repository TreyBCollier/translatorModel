from train import translate
from train import testFunc

import flask


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def index():
    return "Loading translation..."


@app.route('/translateSent/<sentence>')
def translateSent(sentence):
    return translate(sentence)
