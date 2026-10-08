#!/usr/bin/env python3
"""
app.py — Talking Avatar as a desktop app.

    python app.py              opens the app in its own window
    python app.py --browser    in a normal browser tab instead

Or double-click start.bat (Windows) / run ./start.sh (macOS, Linux), which
use the .venv that setup.sh creates. The app needs nothing beyond what
setup.sh installs: Gradio already comes with Chatterbox.

It is a front end for make_avatar.py: every video is a make_avatar.py run in
the background, so anything the app makes can also be made from the command
line, and the other way round. Everything stays on this computer; the app
only listens on 127.0.0.1 unless you pass --listen.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Before Gradio is imported: no usage statistics, and uploads (faces and
# voices) go to a cache inside this folder that is emptied on restart,
# instead of the system temp folder.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".app_cache" / "uploads"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Talking Avatar desktop app.")
    parser.add_argument("--port", type=int, help="Port to use (default: the first free one from 7860)")
    parser.add_argument("--browser", action="store_true",
                        help="Open in a normal browser tab instead of an app window")
    parser.add_argument("--no-open", action="store_true", help="Don't open anything; just print the address")
    parser.add_argument("--listen", action="store_true",
                        help="Also accept connections from other computers on your network. Anyone who "
                             "can reach this computer could then use the app and see your videos, so "
                             "combine it with --auth")
    parser.add_argument("--auth", metavar="USER:PASSWORD", help="Require this login to open the app")
    args = parser.parse_args()

    auth = None
    if args.auth:
        user, _, password = args.auth.partition(":")
        if not (user and password):
            parser.error("--auth needs USER:PASSWORD")
        auth = (user, password)

    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.exit("Gradio isn't installed in this Python environment. Run ./setup.sh once, then start the "
                 "app with start.bat (Windows) or ./start.sh (macOS, Linux).")

    from ui import storage, web
    from ui.jobs import JobManager
    from ui.layout import ICON_PATH, build_app

    if args.listen and not auth:
        print("WARNING: --listen without --auth: anyone on your network can use this app and see your videos.")

    for folder in (storage.OUTPUT_DIR, storage.AVATARS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    web.drop_external_tags()
    manager = JobManager()
    demo = build_app(manager)
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        auth=auth,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        quiet=True,
        show_error=True,
        footer_links=[],
        favicon_path=str(ICON_PATH),
        allowed_paths=[str(storage.OUTPUT_DIR), str(storage.AVATARS_DIR)],
        enable_monitoring=False,
        mcp_server=False,
        ssr_mode=False,
        pwa=True,
        strict_cors=True,
        theme=web.theme(),
        css_paths=[ROOT / "ui" / "style.css"],
        app_kwargs=web.app_kwargs(),
    )
    url = f"http://127.0.0.1:{demo.server_port}/"
    print(f"\nTalking Avatar is running at {url}")
    print("Keep this window open while you use the app. Press Ctrl+C here to quit.\n")
    if not args.no_open:
        web.open_window(url, as_tab=args.browser)
    try:
        demo.block_thread()
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
