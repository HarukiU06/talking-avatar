"""The app's screens: Create, Library, Voice Lab and Setup.

Nothing here renders anything itself. Buttons put jobs on the queue
(ui/jobs.py) and a one-second timer reads the queue back into the page, so a
render keeps going, and shows up again, after the window is closed and
reopened.
"""
from __future__ import annotations

import html
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from ui import pipeline, storage, system
from ui.jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING, JobManager

HERE = Path(__file__).resolve().parent
ICON_PATH = HERE / "icon.svg"
NEW_AVATAR = "New avatar"
esc = html.escape

VOICE_TEST_TEXT = "Hello! This is a quick test of how my cloned voice sounds. Which one is most like me?"
SCRIPT_PLACEHOLDER = ("Type what your avatar should say.\n\n"
                      "Tip: tick \"One video per paragraph\" to make a separate video from each "
                      "paragraph in one go.")


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


# Gradio shows notification text as HTML, so anything not written here
# (names, file paths, error text) is escaped before it goes into one.

def notify(*lines: str) -> None:
    gr.Info("<br>".join(esc(line) for line in lines))


def refuse(*lines: str, title: str = "Not ready yet"):
    """Stop an event with a message for the user (not a bug, so no traceback in the console)."""
    raise gr.Error("<br>".join(esc(line) for line in lines), title=title, print_exception=False)


def _avatar_choices() -> list:
    return [NEW_AVATAR] + [a.name for a in storage.list_avatars()]


def _confirm_js(question: str, skip_value: str | None = None) -> str:
    """A browser confirm() in front of a click. Cancelling sends None to the handler."""
    skip = f"value === {json.dumps(skip_value)} || " if skip_value else ""
    return (f"(value) => (!value || {skip}!confirm({json.dumps(question)}.replace('%s', () => value)))"
            " ? null : value")


# --- HTML pieces -----------------------------------------------------------------

_ICON_SVG: list = []


def _icon_svg() -> str:
    if not _ICON_SVG:
        _ICON_SVG.append(ICON_PATH.read_text(encoding="utf-8"))
    return _ICON_SVG[0]


def _section(step: str, title: str, text: str) -> str:
    return (f'<div class="section-head"><span class="step">{step}</span>'
            f'<div><h3>{title}</h3><p>{text}</p></div></div>')


def header_html(running, waiting: int, gpu: str) -> str:
    if running:
        state = '<span class="chip busy"><span class="dot"></span>Rendering'
        state += f" · {waiting} waiting</span>" if waiting else "</span>"
    else:
        state = '<span class="chip"><span class="dot"></span>Idle</span>'
    return (f'<div class="app-header"><div class="brand">{_icon_svg()}<div>'
            '<h1>Talking Avatar</h1><p>Type it, and watch yourself say it.</p></div></div>'
            f'<div class="header-chips">{state}<span class="chip">{esc(gpu)}</span></div></div>')


def status_html(job, waiting: int) -> str:
    if job is None:
        return ('<div class="status"><div class="status-title">Ready when you are</div>'
                '<div class="status-sub">Choose a face and a voice, type a script and press '
                '<b>Generate video</b>. Rendering runs in the background: you can queue several '
                'videos, and closing this window doesn\'t stop them.</div></div>')
    keys = [s.key for s in job.stages]
    current = keys.index(job.stage) if job.stage in keys else -1
    pills = []
    for i, stage in enumerate(job.stages):
        if job.status == DONE or i < current:
            cls = "done"
        elif i == current and job.status == RUNNING:
            cls = "active"
        elif i == current and job.status == FAILED:
            cls = "failed"
        else:
            cls = ""
        pills.append(f'<li class="{cls}">{esc(stage.label)}</li>')
    title = {
        RUNNING: f"{'Testing voices' if job.kind == 'voices' else 'Rendering'} · {_duration(job.elapsed)}",
        QUEUED: "Waiting to start",
        DONE: f"Finished in {_duration(job.elapsed)}",
        FAILED: "This one didn't work",
        CANCELLED: "Cancelled",
    }[job.status]
    parts = [f'<div class="status-title">{title}</div>',
             f'<div class="status-sub">{esc(job.title)}</div>',
             f'<ul class="stages">{"".join(pills)}</ul>']
    if job.status == RUNNING:
        if job.percent is None:
            parts.append('<div class="bar indeterminate"><div></div></div>')
        else:
            parts.append(f'<div class="bar"><div style="width:{job.percent:.0f}%"></div></div>')
        activity = job.detail or f"{job.stage_label()}..."
        percent = "" if job.percent is None else f"{job.percent:.0f}%"
        parts.append(f'<div class="status-meta"><span>{esc(activity)}</span><span>{percent}</span></div>')
    elif job.status == FAILED:
        parts.append(f'<div class="status-error">{esc(job.error)}</div>'
                     '<div class="status-hint">The full log is under "Details" below.</div>')
    if waiting:
        parts.append(f'<div class="status-foot">{waiting} more waiting in the queue</div>')
    return f'<div class="status {job.status}">{"".join(parts)}</div>'


_QUEUE_STATE = {RUNNING: "Rendering", QUEUED: "Waiting", DONE: "Done", FAILED: "Failed",
                CANCELLED: "Cancelled"}


def _queue_order(jobs: list) -> list:
    active = [j for j in jobs if j.status in (RUNNING, QUEUED)]
    active.sort(key=lambda j: (j.status != RUNNING, j.created))
    finished = sorted((j for j in jobs if j not in active), key=lambda j: j.finished or 0, reverse=True)
    return (active + finished)[:12]


def queue_signature(jobs: list) -> tuple:
    return tuple((j.id, j.status) for j in _queue_order(jobs))


def queue_html(jobs: list) -> str:
    shown = _queue_order(jobs)
    if not shown:
        return '<div class="queue-empty">Nothing queued yet.</div>'
    rows = []
    for job in shown:
        state = _QUEUE_STATE[job.status]
        if job.status == DONE:
            state += f" · {_duration(job.elapsed)}"
        rows.append(f'<li class="q-{job.status}"><span class="q-dot"></span>'
                    f'<span class="q-title">{esc(job.title)}</span><span class="q-state">{state}</span></li>')
    return f'<ul class="queue">{"".join(rows)}</ul>'


def engine_info_md(engine: str, parts: dict) -> str:
    name, model = pipeline.ENGINES[engine]
    missing = [parts[k] for k in pipeline.needed_parts({"engine": engine}) if k in parts and not parts[k].ready
               and k not in ("chatterbox", "ffmpeg")]
    install = ("✓ Installed." if not missing else
               "Not installed yet: run " + " and ".join(f"`{p.setup}`" for p in missing) + ".")
    return f"**{name} ({model}).** {pipeline.ENGINE_INFO[engine]} {install}"


def setup_html(parts: dict, gpu: tuple) -> str:
    has_gpu, gpu_name, gpu_use = gpu
    gpu_text = f"{gpu_name} ({gpu_use})" if gpu_use else gpu_name
    tiles = [
        ("Graphics card", gpu_text, "ok" if has_gpu else "warn"),
        ("PyTorch", system.torch_version(), "ok"),
        ("Free disk space", f"{system.free_disk_gb():.0f} GB", "ok" if system.free_disk_gb() > 20 else "warn"),
    ]
    tile_html = "".join(f'<div class="tile {cls}"><div class="tile-label">{label}</div>'
                        f'<div class="tile-value">{esc(value)}</div></div>' for label, value, cls in tiles)
    rows = []
    for part in parts.values():
        mark = '<span class="ok">✓</span>' if part.ready else '<span class="missing">–</span>'
        todo = "Installed" if part.ready else f"<code>{esc(part.setup)}</code>"
        rows.append(f"<tr><td>{mark}</td><td><b>{esc(part.name)}</b></td>"
                    f"<td>{esc(part.purpose)}</td><td>{todo}</td></tr>")
    return (f'<div class="setup"><div class="tiles">{tile_html}</div>'
            '<table class="parts"><thead><tr><th></th><th>Component</th><th>What it does</th>'
            f'<th>Install</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="hint">Run the setup scripts from this folder in a terminal (on Windows: Git Bash), '
            'then press <b>Check again</b>. You only need the engines you plan to use.</p></div>')


def voice_results_html(folder: Path | None, waiting: bool = False) -> str:
    if folder is None:
        text = ("Generating the voices. They appear here when all of them are done."
                if waiting else "Results appear here: the same sentence in each voice setting.")
        return f'<div class="vl-empty">{text}</div>'
    cards, skipped = [], []
    for filename, (label, _) in pipeline.VOICE_VARIANTS.items():
        clip = Path(folder) / filename
        if not clip.exists():
            skipped.append(label)
            continue
        # Served by Gradio's file route, which only answers for the output and
        # avatar folders the app allows.
        src = "gradio_api/file=" + quote(clip.resolve().as_posix(), safe="/:")
        cards.append(f'<div class="vl-card"><div class="vl-label">{esc(label)}</div>'
                     f'<audio controls preload="metadata" src="{esc(src)}"></audio></div>')
    note = ""
    if skipped:
        note = ('<p class="vl-skipped">Skipped: ' + esc(" · ".join(skipped)) + ". These need XTTS-v2 or "
                "Seed-VC; the Setup tab shows how to install them.</p>")
    return f'<div class="vl-grid">{"".join(cards)}</div>{note}'


PRIVACY_HTML = """
<div class="privacy">
  <h3>Your face and voice stay on this computer</h3>
  <ul>
    <li>Everything is processed locally. The app only listens on this computer (127.0.0.1) unless you start it with <code>--listen</code>.</li>
    <li>The page is locked down so it can't load from or send anything to other websites, and Gradio's usage statistics are switched off.</li>
    <li>Saved avatars are in <code>avatars/</code>, videos in <code>output/</code>. Uploads wait in <code>.app_cache/</code>, which is emptied when the app restarts.</li>
    <li>Only use a face and voice that are yours, or that you have permission to use, and tell people the videos are generated. See README section 7.</li>
  </ul>
</div>
"""


# --- the app -------------------------------------------------------------------------

def build_app(manager: JobManager) -> gr.Blocks:
    shared = {"parts": system.check_parts(), "gpu": system.gpu_summary()}

    with gr.Blocks(title="Talking Avatar", analytics_enabled=False,
                   delete_cache=(6 * 3600, 3 * 24 * 3600)) as demo:
        view = gr.State({})  # what this browser tab currently shows; written only by tick()
        voice_job_id = gr.State(None)  # the Voice Lab test this tab started
        header = gr.HTML(header_html(None, 0, shared["gpu"][1]), elem_id="app-header")

        with gr.Tabs(elem_id="main-tabs") as tabs:
            # ---------------------------------------------------------------- Create
            with gr.Tab("Create", id="create"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, min_width=380):
                        with gr.Column(variant="panel", elem_classes="card"):
                            gr.HTML(_section("1", "Your avatar",
                                             "A face and a voice. Save them once and reuse them every time."))
                            with gr.Row(equal_height=False, elem_classes="align-end"):
                                avatar_dd = gr.Dropdown([NEW_AVATAR], value=NEW_AVATAR, label="Avatar",
                                                        scale=4, elem_id="avatar-select")
                                delete_avatar_btn = gr.Button("Delete", size="sm", scale=1, min_width=90,
                                                              visible=False, elem_id="delete-avatar")
                            face_kind = gr.Radio([("Photo", "photo"), ("Video of me (best results)", "video")],
                                                 value="photo", label="Face from", elem_id="face-kind")
                            with gr.Row(equal_height=False):
                                with gr.Column(min_width=240):
                                    photo = gr.Image(type="filepath", format="png", label="Photo of your face",
                                                     sources=["upload", "webcam", "clipboard"], height=260,
                                                     elem_id="photo")
                                    video = gr.Video(label="Video of you, not talking (1-2 min)", format="mp4",
                                                     sources=["upload", "webcam"], height=260, visible=False,
                                                     elem_id="face-video")
                                with gr.Column(min_width=240):
                                    voice = gr.Audio(type="filepath", format="wav",
                                                     sources=["upload", "microphone"],
                                                     label="Voice sample: 25-30 s of you speaking",
                                                     elem_id="voice")
                            with gr.Accordion("Longer recording of your voice (optional)", open=False):
                                gr.Markdown("Used by *Match my voice more closely*. It only has to show what "
                                            "you sound like, so it can be longer and less clean than the "
                                            "sample above.", elem_classes="note")
                                vc_target = gr.Audio(type="filepath", format="wav",
                                                     sources=["upload", "microphone"], show_label=False,
                                                     elem_id="vc-target")
                            with gr.Row(equal_height=True):
                                avatar_name = gr.Textbox(show_label=False, placeholder="Name this avatar, e.g. Me",
                                                         scale=4, elem_id="avatar-name")
                                save_avatar_btn = gr.Button("Save avatar", scale=1, min_width=120,
                                                            elem_id="save-avatar")

                        with gr.Column(variant="panel", elem_classes="card"):
                            gr.HTML(_section("2", "Script", "What your avatar says, in any of 23 languages."))
                            text = gr.Textbox(show_label=False, lines=6, max_lines=24,
                                              placeholder=SCRIPT_PLACEHOLDER, elem_id="script")
                            with gr.Row(equal_height=True):
                                lang = gr.Dropdown(pipeline.languages("chatterbox"), value=pipeline.default("lang"),
                                                   label="Language", min_width=200, elem_id="lang")
                                per_paragraph = gr.Checkbox(label="One video per paragraph", value=False,
                                                            min_width=240, elem_id="per-paragraph")

                        settings: dict = {}  # parser dest -> component; everything restorable lives here
                        settings["lang"] = lang

                        with gr.Column(variant="panel", elem_classes="card"):
                            gr.HTML(_section("3", "Style", "How the video is made."))
                            settings["engine"] = engine = gr.Radio(
                                [(f"{name}", key) for key, (name, _) in pipeline.ENGINES.items()],
                                value=pipeline.default("engine"), label="Video engine", elem_id="engine")
                            engine_info = gr.Markdown(engine_info_md(pipeline.default("engine"), shared["parts"]),
                                                      elem_classes="note")
                            settings["tts"] = tts = gr.Radio(
                                [("Chatterbox (23 languages)", "chatterbox"),
                                 ("XTTS-v2 (17 languages, non-commercial)", "xtts")],
                                value=pipeline.default("tts"), label="Voice model", elem_id="tts")
                            settings["voice_convert"] = gr.Checkbox(
                                label="Match my voice more closely (Seed-VC)", value=False, elem_id="voice-convert",
                                info="The biggest improvement if the voice doesn't sound like you. "
                                     "Needs setup_seedvc.sh.")

                            with gr.Accordion("Advanced settings", open=False, elem_id="advanced"):
                                gr.Markdown("**Voice**")
                                with gr.Row():
                                    settings["cfg_weight"] = gr.Slider(
                                        0, 1, step=0.05, value=pipeline.default("cfg_weight"),
                                        label="Accent carry-over",
                                        info="Lower (0-0.3) if the voice keeps your sample's accent.")
                                    settings["exaggeration"] = gr.Slider(
                                        0.25, 2, step=0.05, value=pipeline.default("exaggeration"),
                                        label="Expressiveness", info="Higher is livelier, and also faster.")
                                with gr.Row():
                                    settings["pause_ms"] = gr.Slider(
                                        0, 1500, step=50, value=pipeline.default("pause_ms"),
                                        label="Pause between sentences (ms)",
                                        info="0 lets the voice pace itself, which sounds most natural.")
                                    settings["vc_denoise"] = gr.Slider(
                                        0, 1, step=0.05, value=pipeline.default("vc_denoise"),
                                        label="Noise removal (voice match)",
                                        info="Raise if background noise gets through.")
                                with gr.Row():
                                    settings["vc_steps"] = gr.Slider(
                                        10, 50, step=1, value=pipeline.default("vc_steps"),
                                        label="Voice match quality steps")
                                    settings["no_voice_prep"] = gr.Checkbox(
                                        label="Use the voice sample exactly as recorded", value=False,
                                        info="Normally it's trimmed and levelled first.")

                                with gr.Column(visible=True) as ls_box:
                                    gr.Markdown("**Standard engine**")
                                    with gr.Row():
                                        settings["motion"] = gr.Radio(
                                            [("Natural head motion", "idle"), ("Keep the head still", "none")],
                                            value=pipeline.default("motion"), label="From a photo")
                                        settings["latentsync_res"] = gr.Radio(
                                            [256, 512], value=pipeline.default("latentsync_res"),
                                            label="Mouth resolution", info="256 matches the installed model.")
                                    motion_video = gr.Video(label="Your own head-motion clip (optional, 15-20 s)",
                                                            format="mp4", sources=["upload"], height=200)
                                    with gr.Row():
                                        settings["motion_scale"] = gr.Slider(
                                            0.8, 1.5, step=0.05, value=pipeline.default("motion_scale"),
                                            label="Motion strength",
                                            info="Above ~1.2 the face starts to drift from yours.")
                                        settings["inference_steps"] = gr.Slider(
                                            10, 50, step=1, value=pipeline.default("inference_steps"),
                                            label="Lip-sync quality steps")
                                        settings["guidance_scale"] = gr.Slider(
                                            1, 3, step=0.1, value=pipeline.default("guidance_scale"),
                                            label="Lip-sync strength", info="Too high can jitter.")

                                with gr.Column(visible=False) as it_box:
                                    gr.Markdown("**Expressive engine**")
                                    settings["scene_prompt"] = gr.Textbox(
                                        value=pipeline.default("scene_prompt"), lines=3,
                                        label="Scene description",
                                        info="Shapes expression and head motion: tone, setting, body language.")
                                    with gr.Row():
                                        settings["infinitetalk_accel"] = gr.Radio(
                                            [("Fast", "lightx2v"), ("Full quality (hours)", "none")],
                                            value=pipeline.default("infinitetalk_accel"), label="Speed")
                                        settings["infinitetalk_size"] = gr.Radio(
                                            pipeline.choices("infinitetalk_size"),
                                            value=pipeline.default("infinitetalk_size"), label="Resolution (p)")
                                        settings["infinitetalk_mode"] = gr.Radio(
                                            pipeline.choices("infinitetalk_mode"),
                                            value=pipeline.default("infinitetalk_mode"), label="Mode")
                                    with gr.Row():
                                        settings["infinitetalk_steps"] = gr.Slider(
                                            0, 50, step=1, value=0, label="Steps (0 = automatic)")
                                        settings["infinitetalk_seed"] = gr.Number(
                                            value=pipeline.default("infinitetalk_seed"), precision=0,
                                            label="Seed", info="Change it to get a different take.")
                                    with gr.Row():
                                        settings["infinitetalk_quant"] = gr.Radio(
                                            pipeline.choices("infinitetalk_quant"),
                                            value=pipeline.default("infinitetalk_quant"), label="Model precision")
                                        settings["infinitetalk_no_low_vram"] = gr.Checkbox(
                                            label="Keep the whole model on the GPU", value=False,
                                            info="Only with far more than 12 GB of VRAM.")

                                with gr.Column(visible=False) as st_box:
                                    gr.Markdown("**Lightweight engine**")
                                    with gr.Row():
                                        settings["size"] = gr.Radio(pipeline.choices("size"),
                                                                    value=pipeline.default("size"),
                                                                    label="Resolution")
                                        settings["preprocess"] = gr.Dropdown(
                                            pipeline.choices("preprocess"), value=pipeline.default("preprocess"),
                                            label="Framing")
                                    with gr.Row():
                                        settings["expression_scale"] = gr.Slider(
                                            0.5, 2, step=0.05, value=pipeline.default("expression_scale"),
                                            label="Mouth movement")
                                        settings["pose_style"] = gr.Slider(
                                            0, 45, step=1, value=pipeline.default("pose_style"),
                                            label="Head pose style")
                                    with gr.Row():
                                        settings["sadtalker_motion"] = gr.Checkbox(label="Allow head movement")
                                        settings["enhancer"] = gr.Checkbox(label="Sharpen the face (GFPGAN)")
                                        settings["refine_lipsync"] = gr.Checkbox(label="Refine lips (Wav2Lip)")

                                reset_btn = gr.Button("Reset advanced settings", size="sm")

                    with gr.Column(scale=5, min_width=340, elem_id="side"):
                        generate_btn = gr.Button("Generate video", variant="primary", size="lg",
                                                 elem_id="generate")
                        with gr.Column(variant="panel", elem_classes="card"):
                            status = gr.HTML(status_html(None, 0), elem_id="status")
                            with gr.Row():
                                cancel_btn = gr.Button("Cancel this video", size="sm", variant="stop",
                                                       visible=False, elem_id="cancel")
                        result = gr.Video(label="Latest video", interactive=False, buttons=["download"],
                                          height=420, elem_id="result")
                        with gr.Accordion("Details", open=False, elem_id="details"):
                            log_box = gr.Textbox(show_label=False, lines=14, max_lines=14, autoscroll=True,
                                                 interactive=False, elem_classes="log", elem_id="log")
                        with gr.Column(variant="panel", elem_classes="card"):
                            gr.HTML('<div class="mini-head">Queue</div>')
                            queue = gr.HTML(queue_html([]), elem_id="queue")
                            with gr.Row():
                                clear_btn = gr.Button("Clear finished", size="sm", elem_id="clear-finished")
                                stop_all_btn = gr.Button("Stop all", size="sm", variant="stop", elem_id="stop-all")

            # ---------------------------------------------------------------- Library
            with gr.Tab("Library", id="library"):
                lib_rows = gr.State([])
                lib_selected = gr.State(None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        library = gr.Dataframe(headers=["Created", "Text", "Style"], datatype=["str"] * 3,
                                               interactive=False, wrap=True, max_height=620,
                                               show_search="search", column_widths=["18%", "52%", "30%"],
                                               elem_id="library")
                        with gr.Row():
                            refresh_lib_btn = gr.Button("Refresh", size="sm")
                            open_output_btn = gr.Button("Open output folder", size="sm")
                    with gr.Column(scale=5):
                        lib_video = gr.Video(label="Select a video", interactive=False, buttons=["download"],
                                             height=380, elem_id="lib-video")
                        lib_text = gr.Textbox(label="Text", interactive=False, lines=3, elem_id="lib-text")
                        lib_info = gr.Markdown(elem_classes="note")
                        with gr.Row():
                            reuse_btn = gr.Button("Use these settings", variant="primary", elem_id="reuse")
                            delete_video_btn = gr.Button("Delete", variant="stop", elem_id="delete-video")

            # ---------------------------------------------------------------- Voice Lab
            with gr.Tab("Voice Lab", id="voices"):
                gr.HTML('<div class="intro"><h3>Find the voice that sounds most like you</h3>'
                        '<p>Speech takes seconds and video takes minutes, so choose the voice here first. '
                        'The same sentence is spoken with every voice setting that is installed; pick the '
                        'best one and press <b>Use this voice</b>.</p></div>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, variant="panel", elem_classes="card"):
                        vl_voice = gr.Audio(type="filepath", format="wav", sources=["upload", "microphone"],
                                            label="Voice sample", elem_id="vl-voice")
                        vl_target = gr.Audio(type="filepath", format="wav", sources=["upload", "microphone"],
                                             label="Longer recording for voice match (optional)")
                        vl_text = gr.Textbox(value=VOICE_TEST_TEXT, lines=2, label="Test sentence",
                                             elem_id="vl-text")
                        vl_lang = gr.Dropdown(pipeline.languages("chatterbox"), value="en", label="Language")
                        vl_btn = gr.Button("Compare voices", variant="primary", elem_id="vl-go")
                        vl_status = gr.HTML(elem_id="vl-status")
                    with gr.Column(scale=6):
                        # Plain <audio> players in one HTML block: Gradio 6 can freeze the page
                        # when it reveals hidden Audio components and fills them in one update.
                        vl_results = gr.HTML(voice_results_html(None), elem_id="vl-results")
                        with gr.Row(equal_height=False, elem_classes="align-end"):
                            vl_pick = gr.Radio([], label="Your favourite", scale=4, elem_id="vl-pick")
                            vl_use = gr.Button("Use this voice", variant="primary", scale=1, min_width=150,
                                               elem_id="vl-use")

            # ---------------------------------------------------------------- Setup
            with gr.Tab("Setup", id="setup"):
                setup = gr.HTML(setup_html(shared["parts"], shared["gpu"]), elem_id="setup")
                with gr.Row():
                    recheck_btn = gr.Button("Check again", size="sm", elem_id="recheck")
                    open_app_btn = gr.Button("Open the app folder", size="sm")
                gr.HTML(PRIVACY_HTML)
                with gr.Row():
                    quit_btn = gr.Button("Quit Talking Avatar", variant="stop", size="sm", scale=0,
                                         min_width=220, elem_id="quit")
                quit_note = gr.HTML()

        setting_keys = list(settings)
        setting_inputs = [settings[k] for k in setting_keys]
        advanced_keys = setting_keys[setting_keys.index("cfg_weight"):]

        # ---- helpers that need the components -------------------------------------

        def coerce(key: str, value):
            """A stored value as an update for its component, or skip it if it no longer fits."""
            comp = settings[key]
            if value is None:
                return gr.skip()
            if isinstance(comp, (gr.Radio, gr.Dropdown)):
                allowed = {c[1] if isinstance(c, (tuple, list)) else c for c in comp.choices}
                if key == "lang":
                    allowed = set(pipeline.LANGUAGE_NAMES)
                return value if value in allowed else gr.skip()
            if isinstance(comp, gr.Slider):
                try:
                    return min(max(float(value), comp.minimum), comp.maximum)
                except (TypeError, ValueError):
                    return gr.skip()
            if isinstance(comp, gr.Checkbox):
                return bool(value)
            return value

        def apply_settings(values: dict, keys=None) -> list:
            keys = keys or setting_keys
            out = []
            for key in keys:
                if key == "lang" and values.get("lang"):
                    tts_value = values.get("tts") or "chatterbox"
                    out.append(gr.update(choices=pipeline.languages(tts_value), value=values["lang"]))
                elif key == "infinitetalk_steps" and key in values:
                    out.append(values[key] or 0)
                else:
                    out.append(coerce(key, values.get(key)))
            return out

        def options_from(face_kind, photo, face_video, voice_file, vc_file, motion_file, script, values):
            options = dict(zip(setting_keys, values))
            options["infinitetalk_steps"] = int(options.get("infinitetalk_steps") or 0) or None
            options.update(photo=photo if face_kind == "photo" else None,
                           video=face_video if face_kind == "video" else None,
                           voice=voice_file, vc_target=vc_file or None,
                           motion_video=motion_file or None, text=script)
            return options

        # ---- events: avatar ----------------------------------------------------------

        def apply_avatar(name):
            avatar = storage.find_avatar(name)
            storage.save_settings({"last_avatar": avatar.name if avatar else NEW_AVATAR})
            if avatar is None:
                return ("photo", None, None, None, None, "", gr.skip(), gr.skip(), gr.update(visible=False))
            return (avatar.face_kind, avatar.photo, avatar.video, avatar.voice, avatar.vc_target, avatar.name,
                    avatar.voice, avatar.vc_target, gr.update(visible=True))

        avatar_dd.change(apply_avatar, avatar_dd,
                         [face_kind, photo, video, voice, vc_target, avatar_name, vl_voice, vl_target,
                          delete_avatar_btn], api_visibility="private")

        face_kind.change(lambda kind: (gr.update(visible=kind == "photo"), gr.update(visible=kind == "video")),
                         face_kind, [photo, video], api_visibility="private")

        def save_avatar(name, kind, photo_file, video_file, voice_file, vc_file):
            if not (photo_file or video_file or voice_file):
                refuse("Add a photo, video or voice sample before saving.")
            try:
                saved = storage.save_avatar(name, face_kind=kind, photo=photo_file, video=video_file,
                                            voice=voice_file, vc_target=vc_file)
            except (ValueError, OSError) as exc:
                refuse(str(exc), title="Couldn't save")
            notify(f'Saved the avatar "{saved.name}".')
            return gr.update(choices=_avatar_choices(), value=saved.name)

        save_avatar_btn.click(save_avatar, [avatar_name, face_kind, photo, video, voice, vc_target], avatar_dd,
                              api_visibility="private")

        def delete_avatar(name):
            if not name or name == NEW_AVATAR:
                return gr.skip()
            storage.delete_avatar(name)
            notify(f'Deleted the avatar "{name}".')
            return gr.update(choices=_avatar_choices(), value=NEW_AVATAR)

        delete_avatar_btn.click(delete_avatar, avatar_dd, avatar_dd, api_visibility="private",
                                js=_confirm_js('Delete the saved avatar "%s"? Its photo, video and voice '
                                               'files will be removed.', NEW_AVATAR))

        # ---- events: style ---------------------------------------------------------------

        def on_engine(value):
            return (engine_info_md(value, shared["parts"]), gr.update(visible=value == "latentsync"),
                    gr.update(visible=value == "infinitetalk"), gr.update(visible=value == "sadtalker"))

        engine.change(on_engine, engine, [engine_info, ls_box, it_box, st_box], api_visibility="private")

        def on_tts(value, current):
            langs = pipeline.languages(value)
            return gr.update(choices=langs, value=current if current in {c for _, c in langs} else "en")

        tts.change(on_tts, [tts, lang], lang, api_visibility="private")

        reset_btn.click(lambda: apply_settings({k: pipeline.default(k) for k in advanced_keys}, advanced_keys),
                        None, [settings[k] for k in advanced_keys], api_visibility="private")

        # ---- events: generate and the queue -------------------------------------------------

        def generate(avatar, kind, photo_file, video_file, voice_file, vc_file, motion_file, script, split,
                     *values):
            options = options_from(kind, photo_file, video_file, voice_file, vc_file, motion_file, script, values)
            shared["parts"] = system.check_parts()
            found = pipeline.problems(options, shared["parts"])
            if found:
                refuse(*(f"• {p}" for p in found))
            texts = pipeline.paragraphs(script) if split else [pipeline.clean_text(script)]
            if any(len(t) > pipeline.MAX_TEXT_CHARS for t in texts):
                refuse(f"That's a lot of text for one video. Keep each video under "
                       f"{pipeline.MAX_TEXT_CHARS} characters (a few minutes of speech): split it into "
                       "paragraphs and tick \"One video per paragraph\".")
            taken = {str(j.output) for j in manager.jobs() if j.status in (QUEUED, RUNNING)}
            for paragraph in texts:
                job = pipeline.video_job({**options, "text": paragraph},
                                         avatar if avatar != NEW_AVATAR else None, taken)
                taken.add(str(job.output))
                manager.submit(job)
            storage.save_settings({"last_avatar": avatar, "options": dict(zip(setting_keys, values))})
            notify(f"{len(texts)} videos added to the queue." if len(texts) > 1 else "Added to the queue.")

        generate_btn.click(generate, [avatar_dd, face_kind, photo, video, voice, vc_target, motion_video, text,
                                      per_paragraph, *setting_inputs], None, api_visibility="private")

        def cancel_current():
            job = manager.running()
            if job:
                manager.cancel(job.id)

        cancel_btn.click(cancel_current, None, None, api_visibility="private")
        clear_btn.click(manager.clear_finished, None, None, api_visibility="private")
        stop_all_btn.click(lambda confirmed: manager.cancel_all() if confirmed else None, stop_all_btn, None,
                           api_visibility="private",
                           js=_confirm_js("Stop the current video and remove everything waiting in the queue?"))

        # ---- the timer: queue -> page -----------------------------------------------

        def library_data():
            videos = storage.list_videos()
            rows = [[v.meta.get("created") or time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created)),
                     v.meta.get("text") or v.path.stem, v.meta.get("summary", "")] for v in videos]
            return rows, [str(v.path) for v in videos]

        def library_signature():
            try:
                entries = [e for e in os.scandir(storage.OUTPUT_DIR) if e.name.endswith(".mp4")]
            except OSError:
                return (0, 0)
            return (len(entries), max((e.stat().st_mtime for e in entries), default=0))

        def tick(seen, voice_job_key):
            seen = dict(seen or {})

            def fresh(key, value, update=lambda v: v):
                if seen.get(key) == value:
                    return gr.skip()
                seen[key] = value
                return update(value)

            jobs = manager.jobs()
            running = next((j for j in jobs if j.status == RUNNING), None)
            waiting = sum(1 for j in jobs if j.status == QUEUED)
            finished = [j for j in jobs if j.finished]
            focus = running or (max(finished, key=lambda j: j.finished) if finished else None)
            done_videos = [j for j in finished if j.kind == "video" and j.status == DONE]
            latest = max(done_videos, key=lambda j: j.finished).output if done_videos else None

            outputs = [
                fresh("header", header_html(running, waiting, shared["gpu"][1])),
                fresh("status", status_html(focus, waiting)),
                fresh("cancel", running is not None, lambda v: gr.update(visible=v)),
                fresh("log", focus.log_tail(200) if focus else ""),
                fresh("queue", queue_signature(jobs), lambda _: queue_html(jobs)),
                fresh("result", str(latest) if latest else None),
            ]
            if seen.get("library") != library_signature():
                seen["library"] = library_signature()
                rows, paths = library_data()
                outputs += [rows, paths]
            else:
                outputs += [gr.skip(), gr.skip()]

            voice_job = manager.get(voice_job_key) if voice_job_key else None
            voice_outputs = [gr.skip()] * 3  # status, results, picker
            if voice_job is not None:
                if voice_job.status in (QUEUED, RUNNING):
                    voice_outputs[0] = fresh("vl_status", ("busy", voice_job.status, voice_job.detail),
                                             lambda _: voice_status_html(voice_job))
                elif seen.get("voice_shown") != voice_job.id:
                    seen["voice_shown"] = voice_job.id
                    voice_outputs[0] = voice_status_html(voice_job)
                    if voice_job.status == DONE:
                        produced = [(label, f) for f, (label, _) in pipeline.VOICE_VARIANTS.items()
                                    if (voice_job.output / f).exists()]
                        voice_outputs[1] = voice_results_html(voice_job.output)
                        voice_outputs[2] = gr.update(choices=produced, value=produced[0][1] if produced else None)
            return [seen, *outputs, *voice_outputs]

        def voice_status_html(job):
            if job.status == QUEUED:
                return '<div class="vl-status">Waiting for the current video to finish...</div>'
            if job.status == RUNNING:
                return (f'<div class="vl-status"><div class="bar indeterminate"><div></div></div>'
                        f'{esc(job.detail or "Starting...")}</div>')
            if job.status == DONE:
                return '<div class="vl-status ok">Done. Listen on the right and pick a favourite.</div>'
            return f'<div class="status-error">{esc(job.error or "Cancelled.")}</div>'

        timer = gr.Timer(1.0)
        timer.tick(tick, [view, voice_job_id],
                   [view, header, status, cancel_btn, log_box, queue, result, library, lib_rows,
                    vl_status, vl_results, vl_pick],
                   show_progress="hidden", api_visibility="private")

        # ---- events: library ---------------------------------------------------------------

        def select_video(paths, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row is None or row >= len(paths):
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()
            path = Path(paths[row])
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) \
                if path.with_suffix(".json").exists() else {}
            details = []
            if meta.get("avatar"):
                details.append(f"**Avatar:** {esc(meta['avatar'])}")
            if meta.get("summary"):
                details.append(f"**Style:** {esc(meta['summary'])}")
            if meta.get("render_seconds"):
                details.append(f"**Rendered in:** {_duration(meta['render_seconds'])}")
            details.append(f"**File:** `{esc(path.name)}`")
            return str(path), str(path), meta.get("text", ""), "  \n".join(details)

        library.select(select_video, lib_rows, [lib_selected, lib_video, lib_text, lib_info],
                       api_visibility="private")
        refresh_lib_btn.click(library_data, None, [library, lib_rows], api_visibility="private")
        open_output_btn.click(lambda: storage.open_folder(storage.OUTPUT_DIR), None, None,
                              api_visibility="private")

        def delete_video(path):
            if not path:
                return [gr.skip()] * 6
            try:
                storage.delete_video(path)
            except ValueError as exc:
                refuse(str(exc), title="Couldn't delete")
            rows, paths = library_data()
            notify("Video deleted.")
            return rows, paths, None, None, "", ""

        delete_video_btn.click(delete_video, lib_selected,
                               [library, lib_rows, lib_selected, lib_video, lib_text, lib_info],
                               api_visibility="private",
                               js=_confirm_js("Delete this video? This can't be undone."))

        def reuse(path):
            if not path:
                refuse("Select a video in the list first.")
            meta = storage._read_json(Path(path).with_suffix(".json"))
            values = meta.get("settings") or {}
            avatar = meta.get("avatar")
            avatar_update = avatar if storage.find_avatar(avatar) else gr.skip()
            notify("Settings loaded into Create.")
            return [gr.Tabs(selected="create"), avatar_update, meta.get("text", gr.skip()),
                    *apply_settings(values)]

        reuse_btn.click(reuse, lib_selected, [tabs, avatar_dd, text, *setting_inputs], api_visibility="private")

        # ---- events: voice lab ---------------------------------------------------------------

        def compare_voices(voice_file, target_file, sentence, language, avatar):
            shared["parts"] = system.check_parts()
            if not voice_file:
                refuse("Add a voice sample first.")
            if not pipeline.clean_text(sentence):
                refuse("Type a test sentence.")
            for key in ("chatterbox", "ffmpeg"):
                part = shared["parts"][key]
                if not part.ready:
                    refuse(f"{part.name} isn't installed yet. Run {part.setup}.")
            label = avatar if avatar != NEW_AVATAR else Path(voice_file).stem
            job = manager.submit(pipeline.voices_job(voice_file, sentence, language, target_file, label))
            return [job.id, voice_status_html(job), voice_results_html(None, waiting=True),
                    gr.update(choices=[], value=None)]

        vl_btn.click(compare_voices, [vl_voice, vl_target, vl_text, vl_lang, avatar_dd],
                     [voice_job_id, vl_status, vl_results, vl_pick], api_visibility="private")

        voice_keys = ["tts", "voice_convert", "cfg_weight", "pause_ms", "no_voice_prep"]

        def use_voice(filename):
            if filename not in pipeline.VOICE_VARIANTS:
                refuse("Compare voices first, then pick your favourite.")
            label, values = pipeline.VOICE_VARIANTS[filename]
            notify(f"Voice set to: {label}.")
            return [gr.Tabs(selected="create"), *apply_settings(values, voice_keys)]

        vl_use.click(use_voice, vl_pick, [tabs, *[settings[k] for k in voice_keys]], api_visibility="private")

        # ---- events: setup ---------------------------------------------------------------

        def recheck(engine_value):
            shared["parts"], shared["gpu"] = system.check_parts(), system.gpu_summary()
            return setup_html(shared["parts"], shared["gpu"]), engine_info_md(engine_value, shared["parts"])

        recheck_btn.click(recheck, engine, [setup, engine_info], api_visibility="private")
        open_app_btn.click(lambda: storage.open_folder(storage.ROOT), None, None, api_visibility="private")

        def quit_app(confirmed):
            if not confirmed:
                return gr.skip()
            manager.shutdown()
            threading.Timer(0.5, demo.close).start()
            return ('<div class="status-error">Talking Avatar has stopped. You can close this window; '
                    'start it again with start.bat or ./start.sh.</div>')

        quit_btn.click(quit_app, quit_btn, quit_note, api_visibility="private",
                       js=_confirm_js("Quit Talking Avatar? A video that is still rendering will be stopped."))

        # ---- first load ---------------------------------------------------------------

        def on_load():
            stored = storage.load_settings()
            choices = _avatar_choices()
            last = stored.get("last_avatar")
            rows, paths = library_data()
            return [gr.update(choices=choices, value=last if last in choices else NEW_AVATAR),
                    *apply_settings(stored.get("options") or {}), rows, paths]

        demo.load(on_load, None, [avatar_dd, *setting_inputs, library, lib_rows], api_visibility="private")
    return demo
