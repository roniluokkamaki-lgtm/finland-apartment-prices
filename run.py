"""
Launcher — Asuntojen neliöhintalaskuri.
Run with: python run.py
"""

import subprocess
import sys
import time
import webbrowser

URL = "http://localhost:8501"

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app/app.py",
     "--server.headless", "true"],
)

# Wait for the server to start, then open the browser
time.sleep(3)
webbrowser.open(URL)

print(f"Laskuri auki osoitteessa {URL}")
print("Sulje painamalla Ctrl+C.")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
