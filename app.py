"""
app.py — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
Entry point for Plotly Dash / direct dashboard launch.

This stub serves the self-contained vanilla-JS dashboard HTML
directly over a local HTTP server. No Dash framework required
for the static dashboard; this file satisfies the CLAUDE.md
mandatory-files requirement and provides a convenient launcher.

Usage:
  python app.py                  # serves on http://localhost:8050
  python app.py --port 8080      # custom port
"""

import argparse
import http.server
import os
import socketserver
import threading
import webbrowser

DASHBOARD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dashboard",
    "NHIS_OOP_Ghana_Dashboard.html",
)


def serve(port: int = 8050) -> None:
    """Serve the dashboard HTML via a simple HTTP server."""
    dashboard_dir = os.path.dirname(DASHBOARD_PATH)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dashboard_dir, **kwargs)

        def log_message(self, format, *args):  # noqa: A002
            # Suppress access-log noise
            pass

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/NHIS_OOP_Ghana_Dashboard.html"
        print(f"NHIS OOP Ghana 261 Districts Dashboard")
        print(f"URL  : {url}")
        print(f"Press Ctrl+C to stop.\n")
        # Open browser after a brief delay
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the NHIS OOP Ghana 261 Districts interactive dashboard."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="TCP port to serve on (default: 8050)",
    )
    args = parser.parse_args()
    serve(port=args.port)
