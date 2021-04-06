# Importing the translate functions
from train import translate
from trainEnglishToFrench import translateEnglishToFrench

# Importing Flask
import flask

# Initialise the Flask app
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Set default route


@app.route('/')
def index():
    return "Loading translation..."

# Set route for French-to-English transltion


@app.route('/translateFrench/<sentence>')
def translateFrench(sentence):
    return translate(sentence)

# Set route for English-to-French transltion


@app.route('/translateEnglish/<sentence>')
def translateEnglish(sentence):
    return translateEnglishToFrench(sentence)
