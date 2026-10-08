"""The browser side: privacy headers, theme, and opening the app window."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import gradio as gr
from starlette.middleware import Middleware

# The page may load from and talk to this app only. Gradio's page template
# pulls a script from cdnjs and its default theme pulls fonts from Google;
# with this policy the browser refuses both, and anything injected into the
# page couldn't send data out either.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "frame-src 'self' blob:",
    "frame-ancestors 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
])
_HEADERS = [
    (b"content-security-policy", CONTENT_SECURITY_POLICY.encode()),
    (b"referrer-policy", b"no-referrer"),
]


class _SecurityHeaders:
    """Plain ASGI middleware, so streamed responses (progress events) pass straight through."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *_HEADERS]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def app_kwargs() -> dict:
    return {"middleware": [Middleware(_SecurityHeaders)]}


def drop_external_tags() -> None:
    """Remove the cdnjs script and Google Fonts preconnect from Gradio's page template.

    The security policy already stops both from loading; this only saves the
    browser the console errors and the preconnect's DNS lookup. If a Gradio
    update moves things around this quietly does nothing.
    """
    try:
        from gradio import routes
        original = routes.templates.TemplateResponse
    except (ImportError, AttributeError):
        return
    external = re.compile(r'\s*<(?:link rel="preconnect"[^>]*https://fonts\.[^>]*>|script\s+src="https://cdnjs\.'
                          r'[^"]*"[^>]*>\s*</script>)', re.S)

    def response(*args, **kwargs):
        resp = original(*args, **kwargs)
        try:
            body = resp.body.decode("utf-8")
            cleaned = external.sub("", body)
            if cleaned != body:
                resp.body = cleaned.encode("utf-8")
                resp.headers["content-length"] = str(len(resp.body))
        except (AttributeError, UnicodeDecodeError):
            pass
        return resp

    routes.templates.TemplateResponse = response


def theme() -> gr.themes.Base:
    system_fonts = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Noto Sans",
                    "Helvetica Neue", "Arial", "sans-serif"]
    mono_fonts = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "Liberation Mono", "monospace"]
    return gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
                          radius_size="lg", font=system_fonts, font_mono=mono_fonts)


def _app_browsers() -> list:
    """Chromium-based browsers, which can open a page as its own app window."""
    found = []
    if sys.platform == "win32":
        for base in filter(None, (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                                  os.environ.get("LOCALAPPDATA"))):
            found += [Path(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      Path(base, "Google", "Chrome", "Application", "chrome.exe")]
    elif sys.platform == "darwin":
        found += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                  Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                  Path("/Applications/Chromium.app/Contents/MacOS/Chromium")]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"):
            path = shutil.which(name)
            if path:
                found.append(Path(path))
    return [p for p in found if p.exists()]


def open_window(url: str, as_tab: bool = False) -> None:
    """Open the app in its own window (Edge/Chrome app mode), or a normal tab."""
    if not as_tab:
        for browser in _app_browsers():
            try:
                subprocess.Popen([str(browser), f"--app={url}", "--window-size=1440,960"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue
    webbrowser.open(url)
