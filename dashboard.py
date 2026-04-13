import os
import csv
import subprocess
from flask import Flask, render_template, jsonify

app = Flask(__name__)

LOG_DIR = "logs"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/launch")
def launch_adas():
    """Spawns the main.py driving system as a background process."""
    try:
        # Popen executes it without blocking the Flask server
        subprocess.Popen(["python", "main.py"])
        return jsonify({"status": "success", "message": "ADAS System Launched! Please check your windows."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/logs")
def get_logs():
    """Returns a list of all session log filenames."""
    if not os.path.exists(LOG_DIR):
        return jsonify([])
    
    files = [f for f in os.listdir(LOG_DIR) if f.endswith('.csv')]
    files.sort(reverse=True) # Newest files first
    return jsonify(files)

@app.route("/api/logs/<filename>")
def get_log_data(filename):
    """Parses a specific CSV file and returns it as a JSON array for charting."""
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
        
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
