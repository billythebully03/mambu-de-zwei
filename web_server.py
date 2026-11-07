"""Небольшой HTTP-сервер для Render.

Web-сервис Render должен слушать открытый порт, иначе деплой падает.
Здесь же отвечают пинги (UptimeRobot), чтобы сервис не засыпал.
"""

import logging
import os
import threading

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
def index():
    return "Мембу на связи.", 200


@app.get("/health")
def health():
    return jsonify(status="ok"), 200


def start_web_server() -> threading.Thread:
    port = int(os.getenv("PORT", "10000"))

    def run() -> None:
        logging.info("Web server listening on port %s", port)
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
        )

    thread = threading.Thread(target=run, name="web", daemon=True)
    thread.start()
    return thread
