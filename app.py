from flask import Flask, render_template, Response, send_from_directory, request, jsonify
from main import run_prompt
import os
import json
import re
import traceback
from utils.stream import mjpeg_generator

app = Flask(__name__, static_url_path='', template_folder="templates")

@app.route('/app_static/<path:filename>')
def serve_app_static(filename):
    static_dir = os.path.join(os.getcwd(), 'app_static')
    return send_from_directory(static_dir, filename)

@app.route('/static/<path:path>')
def serve_static_files(path):
    print(f"[DEBUG] Serving static file: /static/{path}")
    return send_from_directory('static', path)

@app.route("/")
def home():
    """Serves the main index.html file."""
    return send_from_directory("templates", "index.html")

@app.route("/stream")
def stream():
    # Live MJPEG stream for the <img src="/stream">
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/run", methods=["POST"])
def execute_test_run_route():
    """Executes a full test run based on a prompt and user credentials."""
    data = request.json
    prompt = data.get("prompt")


    
    final_prompt = f"{prompt}"

    try:
        result = run_prompt(final_prompt)
        response = {
            "status": "success",
            "result": result.get("text", "No text result returned."),
            "test_status": result.get("status", "unknown")
        }

        if result.get("gif_path"):
            response["gif_url"] = "/" + result["gif_path"]

        # Add this:
        if result.get("video_path"):
            response["video_url"] = "/" + result["video_path"]


       

        return jsonify(response)
    except Exception as e:
        print(f"[ERROR] Exception in /run: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
