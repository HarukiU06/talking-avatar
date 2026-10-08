"""Background job queue: runs make_avatar.py one job at a time.

Rendering takes minutes (hours with InfiniteTalk), far longer than a browser
request should stay open, so the interface never renders inside an event
handler. It adds a Job to this queue and polls it. One worker thread runs the
jobs in order, because they all need the same GPU, and every job is its own
make_avatar.py process:

- Cancelling has to stop the model processes make_avatar.py starts in their
  own virtual environments, and that only works by killing a process tree.
- A crash or out-of-memory error inside a model ends that job, not the app.
- Every job starts on an empty GPU; nothing stays loaded between jobs.

Closing the browser doesn't stop anything: the queue lives in the app process,
and reopening the page picks the current job back up.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
MAKE_AVATAR = ROOT / "make_avatar.py"

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)

LOG_LINES_KEPT = 400  # per job, in memory; the whole log is written to a file


@dataclass(frozen=True)
class Stage:
    key: str
    label: str


@dataclass
class Job:
    kind: str            # "video" or "voices"
    title: str           # short label for the queue
    args: list           # make_avatar.py arguments
    stages: list         # Stage objects, in the order they run
    output: Path         # the video, or the folder the voice clips go to
    log_path: Path
    meta: dict = field(default_factory=dict)
    on_done: Callable | None = field(default=None, repr=False)  # called with the job on success
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    stage: str | None = None
    percent: float | None = None   # progress of the current step, when a tool reports it
    detail: str = ""                # what is happening right now, in a few words
    error: str = ""
    lines: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES_KEPT), repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    proc: subprocess.Popen | None = field(default=None, repr=False)
    cancel_requested: bool = False
    _last_was_progress: bool = field(default=False, repr=False)
    _variants_seen: int = field(default=0, repr=False)

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def log_tail(self, n: int = 200) -> str:
        with self.lock:
            lines = list(self.lines)[-n:]
        return "\n".join(lines)

    def stage_label(self) -> str:
        for stage in self.stages:
            if stage.key == self.stage:
                return stage.label
        return ""


# --- reading make_avatar.py's output -----------------------------------------

# Lines make_avatar.py prints when a step starts, mapped to the stage keys
# planned in ui/pipeline.py. Matching on output keeps make_avatar.py free of
# any app-specific progress protocol; a line that stops matching only costs
# the progress display, never the render.
STAGE_MARKERS = (
    ("Prepared voice reference", "speech"),
    ("Loading Chatterbox", "speech"),
    ("Running XTTS:", "speech"),
    ("Seed-VC reference:", "convert"),
    ("Running Seed-VC:", "convert"),
    ("Source video", "motion"),
    ("Running LivePortrait:", "motion"),
    ("Running LatentSync:", "lipsync"),
    ("Running SadTalker:", "render"),
    ("Running InfiniteTalk:", "render"),
    ("Running Wav2Lip refinement:", "refine"),
)

# tqdm ("45%|####  | 9/20 [00:30<00:37, 3.40s/it]") and rich ("━━━ 45%") bars.
# A percentage alone isn't enough: plenty of ordinary log lines contain one.
_PROGRESS_HINT = re.compile(r"%\||━|\d\s*it/s|s/it")
_PERCENT = re.compile(r"(?<![\d.])(\d{1,3})(?:\.\d+)?%")
_TQDM_COUNTS = re.compile(r"\|\s*(\d+/\d+\s*\[[^\]]*\])")


def parse_percent(line: str) -> float | None:
    if not _PROGRESS_HINT.search(line):
        return None
    match = _PERCENT.search(line)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0 <= value <= 100 else None


# Known failures, translated into what to do about them. Checked against the
# end of the log, most specific first.
_HINTS = (
    (re.compile(r"Couldn't find a Python executable in venv \S*?(\.venv-[\w-]+)"),
     "Part of the pipeline isn't installed: its environment {0} is missing. "
     "Open the Setup tab to see which setup script to run."),
    (re.compile(r"Couldn't find (LatentSync|LivePortrait|SadTalker|Seed-VC|Wav2Lip|InfiniteTalk) at"),
     "{0} isn't installed yet. Open the Setup tab to see how to install it."),
    (re.compile(r"Missing InfiniteTalk checkpoints|Missing the lightx2v LoRA"),
     "InfiniteTalk's model files are missing. Re-run ./setup_infinitetalk.sh."),
    (re.compile(r"Couldn't find .*wav2lip_gan\.pth"),
     "The Wav2Lip checkpoint has to be downloaded by hand. See README section 3 (Wav2Lip)."),
    (re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED", re.I),
     "The graphics card ran out of memory. Close other programs that use the GPU "
     "(games, browsers with hardware acceleration, other renders) and try again, "
     "or use a shorter text."),
    (re.compile(r"appears to be silent"),
     "The voice sample seems to be silent. Record 25-30 seconds of clear speech."),
    (re.compile(r"No such file or directory: '(ffmpeg|ffprobe)'|'(ffmpeg|ffprobe)' is not recognized"),
     "ffmpeg isn't installed or isn't on PATH. See the Setup tab."),
    (re.compile(r"(?:Unsupported language|doesn't support language) '(\w+)'"),
     "The selected voice model can't speak language '{0}'. Pick another language or voice model."),
)
_EXCEPTION_LINE = re.compile(r"^(?:[\w.]+(?:Error|Exception)|error)\b:?\s*(.*)", re.I)


def explain_failure(lines: list) -> str:
    tail = [line.strip() for line in lines[-80:] if line.strip()]
    text = "\n".join(tail)
    for pattern, message in _HINTS:
        match = pattern.search(text)
        if match:
            return message.format(*[g for g in match.groups() if g] or [""])
    for line in reversed(tail):
        if _EXCEPTION_LINE.match(line) or ": error:" in line:
            return line
    return tail[-1] if tail else "The process stopped without any output."


# --- processes -----------------------------------------------------------------

def _process_group_kwargs() -> dict:
    # POSIX: a new session makes the job its own process group, so the whole
    # tree (make_avatar.py plus the model processes it starts) can be signalled
    # at once. Windows gets nothing extra: taskkill /T walks the tree itself,
    # and staying in the console's group means closing the console also ends
    # the job instead of leaving it running unseen.
    return {} if os.name == "nt" else {"start_new_session": True}


def kill_tree(proc: subprocess.Popen, grace_s: float = 8.0) -> None:
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:  # anything that ignored SIGTERM, children included
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def display_command(cmd: list) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


class JobManager:
    """Owns the queue and the worker thread. One instance per app."""

    def __init__(self, root: Path = ROOT, work_dir: Path | None = None,
                 python: str | None = None, script: Path | None = None):
        self.root = Path(root)
        self.work_dir = Path(work_dir or self.root / ".app_cache" / "jobs")
        self.python = python or sys.executable
        self.script = Path(script or MAKE_AVATAR)
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        shutil.rmtree(self.work_dir, ignore_errors=True)  # left over from a previous run
        self._thread = threading.Thread(target=self._loop, name="render-queue", daemon=True)
        self._thread.start()

    # -- used by the interface --

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs.append(job)
        self._wake.set()
        return job

    def jobs(self) -> list:
        with self._lock:
            return list(self._jobs)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs if j.id == job_id), None)

    def running(self) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs if j.status == RUNNING), None)

    def waiting(self) -> list:
        with self._lock:
            return [j for j in self._jobs if j.status == QUEUED]

    def cancel(self, job_id: str) -> None:
        proc = None
        with self._lock:
            job = next((j for j in self._jobs if j.id == job_id), None)
            if job is None or job.status in FINISHED:
                return
            job.cancel_requested = True
            if job.status == QUEUED:
                job.status, job.finished = CANCELLED, time.time()
                return
            proc = job.proc
        if proc is not None:
            kill_tree(proc)

    def cancel_all(self) -> None:
        for job in self.jobs():
            if job.status == QUEUED:
                self.cancel(job.id)
        current = self.running()
        if current:
            self.cancel(current.id)

    def clear_finished(self) -> None:
        with self._lock:
            self._jobs = [j for j in self._jobs if j.status not in FINISHED]

    def shutdown(self) -> None:
        self._stopping = True
        self.cancel_all()
        self._wake.set()

    # -- worker --

    def _loop(self) -> None:
        while not self._stopping:
            job = self._take_next()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception as exc:  # never let one job take the queue down
                self._finish(job, FAILED, f"The app hit an internal error: {exc}")

    def _take_next(self) -> Job | None:
        with self._lock:
            for job in self._jobs:
                if job.status == QUEUED:
                    job.status, job.started = RUNNING, time.time()
                    job.stage = job.stages[0].key if job.stages else None
                    return job
        return None

    def _run(self, job: Job) -> None:
        # Temp files of a cancelled or crashed run would otherwise stay in the
        # system temp folder, and they are copies of someone's face and voice.
        # Pointing TMP at a per-job folder lets them all go with it, children
        # included.
        tmp = self.work_dir / job.id
        tmp.mkdir(parents=True, exist_ok=True)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
        cmd = [self.python, "-u", str(self.script), *job.args]
        try:
            with open(job.log_path, "w", encoding="utf-8", buffering=1) as log:
                log.write(f"$ {display_command(cmd)}\n\n")
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", bufsize=1,
                        **_process_group_kwargs())
                except OSError as exc:
                    self._finish(job, FAILED, f"Couldn't start make_avatar.py: {exc}")
                    return
                with job.lock:
                    job.proc = proc
                if job.cancel_requested:  # cancelled between leaving the queue and starting
                    kill_tree(proc)
                last_logged_percent = None
                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    percent = self._consume(job, line)
                    # Progress bars redraw many times a second; keep one line
                    # per whole percent in the file.
                    if percent is None or int(percent) != last_logged_percent:
                        log.write(line + "\n")
                    last_logged_percent = None if percent is None else int(percent)
                returncode = proc.wait()
        finally:
            with job.lock:
                job.proc = None
            shutil.rmtree(tmp, ignore_errors=True)

        if job.cancel_requested:
            self._finish(job, CANCELLED, "Cancelled.")
        elif returncode != 0:
            with job.lock:
                lines = list(job.lines)
            self._finish(job, FAILED, explain_failure(lines))
        elif not job.output.exists():
            self._finish(job, FAILED, "make_avatar.py finished but didn't write "
                                      f"{job.output.name}. Check the log.")
        else:
            self._finish(job, DONE)

    def _consume(self, job: Job, line: str) -> float | None:
        if not line.strip():
            return None  # mostly the empty line a progress bar's first \r leaves behind
        percent = parse_percent(line)
        with job.lock:
            # A redrawn progress bar replaces its previous line instead of
            # adding one per redraw.
            if percent is not None and job._last_was_progress and job.lines:
                job.lines[-1] = line
            else:
                job.lines.append(line)
            job._last_was_progress = percent is not None
            if percent is not None:
                job.percent = percent
                counts = _TQDM_COUNTS.search(line)
                if counts:
                    job.detail = counts.group(1)
                return percent
            if job.kind == "voices" and line.startswith("--- "):
                job._variants_seen += 1
                total = job.meta.get("variants", 8)
                job.percent = 100.0 * (job._variants_seen - 1) / total
                job.detail = f"Voice {job._variants_seen} of {total}: " + line[4:].split(":")[0]
                return None
            for marker, key in STAGE_MARKERS:
                if line.startswith(marker):
                    self._advance(job, key)
                    break
        return None

    @staticmethod
    def _advance(job: Job, key: str) -> None:
        keys = [s.key for s in job.stages]
        if key not in keys:
            return
        if job.stage in keys and keys.index(key) < keys.index(job.stage):
            return  # stages only move forward
        if key != job.stage:
            job.stage, job.percent, job.detail = key, None, ""

    def _finish(self, job: Job, status: str, error: str = "") -> None:
        with job.lock:
            job.status, job.error, job.finished = status, error, time.time()
            if status == DONE:
                job.percent, job.detail = 100.0, ""
        if status == DONE and job.on_done:
            try:
                job.on_done(job)
            except Exception as exc:  # the render itself succeeded; don't report it as failed
                with job.lock:
                    job.lines.append(f"(couldn't save the details for the library: {exc})")
