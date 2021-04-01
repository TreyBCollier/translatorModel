from train import translate

import flask


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/translateSent/<sentence>')
def translateSent(sentence):
    return translate(sentence)
