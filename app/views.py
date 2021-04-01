from app import app

from train import translate

from flask import request, render_template


@app.route("/")
def index():
    """
    This route will render a template.
    If a query string comes into the URL, it will return a parsed
    dictionary of the query string keys & values, using request.args
    """

    args = None

    if request.args:

        args = request.args

        args = translate(args)

        return render_template("public/index.html", args=args)

    return render_template("public/index.html", args=args)
