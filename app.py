import os
import json
import argparse
from bottle import get, post, run, request, response
from model import Model

model = Model()


@post('/')
def do_evaluate():
    try:
        data = request.json
    except Exception:
        response.status = 400
        return

    score = model.predict_sample(data['text'])
    response.headers['Content-Type'] = 'application/json'
    return json.dumps({'score': score})


@get('/help')
def get_help():
    return "<a href=https://www.youtube.com/watch?v=JC0tqZfMv34>JC0tqZfMv34</a>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a web app')
    parser.add_argument("--model", '-m', help="Path to the model weights")
    args = parser.parse_args()

    model.load(args.model)
    port = os.environ.get("PORT", 8080)
    run(host = 'localhost', port = port, reloader = True)
