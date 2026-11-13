"""
bg_remover.py
=================
Author: Akascape
License: MIT License - Copyright (c) 2026 Akascape

Background removal for the Rembg Fuse (DaVinci Resolve / Fusion).

Usage:
    bg_remover.py INPUT OUTPUT MODEL   Process one image (handy for testing).
    bg_remover.py --serve SPOOL_DIR    Run the shared worker used by the fuse.

Shared worker protocol (all files live in SPOOL_DIR):
    worker.lock       held while a worker runs, so only one runs at a time
    worker.heartbeat  "<unix time> <pid>", rewritten every second
    worker.log        worker output
    <id>.req          request from the fuse: input path, output path, model
                      (one per line); the fuse renames it into place
    <id>.done         reply: "ok <provider> <seconds>"
    <id>.err          reply: error message

Models are loaded once and kept on the GPU. The worker exits after
IDLE_TIMEOUT seconds without requests, or when this file changes on disk.
"""

import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict

IDLE_TIMEOUT = 15 * 60
MAX_SESSIONS = 2  # loaded models kept in (GPU) memory

# onnxruntime's CoreML backend cannot run these large transformer models in
# practice: building birefnet-general took >10 minutes with the NeuralNetwork
# format and the MLProgram format silently fell back to CPU. Skip CoreML for
# them instead of stalling the first frame.
COREML_UNSUPPORTED_PREFIXES = ("birefnet", "ben2", "bria", "sam")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_output(log_path=None):
    """Send output to log_path, or to a temp log when there is no stdout (pythonw)."""
    if log_path is None and sys.stdout is not None and sys.stderr is not None:
        return
    path = log_path or os.path.join(tempfile.gettempdir(), "rembg_fuse.log")
    stream = open(path, "w" if log_path else "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    # Native libraries (onnxruntime, CUDA, CoreML) write straight to file
    # descriptors 1/2; point those at the log too so their errors are kept.
    for fd in (1, 2):
        try:
            os.dup2(stream.fileno(), fd)
        except OSError:
            pass


# --------------------------------------------------------------------------
# GPU sessions
# --------------------------------------------------------------------------


def gpu_providers(model):
    """onnxruntime providers: the platform's GPU backend first, CPU as fallback."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        # Load CUDA/cuDNN from the nvidia-* pip packages installed by
        # onnxruntime-gpu[cuda,cudnn], so no system-wide CUDA install is needed.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:
                log(f"[WARNING] Could not preload CUDA libraries: {e}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        if model.startswith(COREML_UNSUPPORTED_PREFIXES):
            log(f"[WARNING] CoreML cannot run {model}; using CPU (expect seconds per frame).")
            return ["CPUExecutionProvider"]
        cache = os.path.join(os.path.expanduser("~"), ".rembg", "coreml-cache")
        os.makedirs(cache, exist_ok=True)
        # NeuralNetwork format: fastest for these models in testing; the newer
        # MLProgram format was not faster and fails to build some models.
        options = {
            "ModelFormat": "NeuralNetwork",
            "MLComputeUnits": "ALL",
            "ModelCacheDirectory": cache,
        }
        return [("CoreMLExecutionProvider", options), "CPUExecutionProvider"]
    log(f"[WARNING] No GPU provider available (have {available}); using CPU.")
    return ["CPUExecutionProvider"]


def load_session(model):
    """Create a rembg session on the GPU. Returns (session, active provider)."""
    import rembg

    started = time.time()
    try:
        session = rembg.new_session(model, providers=gpu_providers(model))
    except Exception as e:
        # Also covers rembg's fallback warning crashing on provider options
        # (TypeError) when the GPU backend could not take the model.
        log(f"[WARNING] GPU session for {model} failed ({type(e).__name__}: {e}); retrying on CPU.")
        session = rembg.new_session(model, providers=["CPUExecutionProvider"])

    provider = session.inner_session.get_providers()[0]
    log(f"Loaded {model} on {provider} in {time.time() - started:.1f}s")
    if provider == "CPUExecutionProvider":
        log(f"[WARNING] {model} is running on the CPU, not the GPU.")
    return session, provider


def remove_background(session, input_path, output_path):
    import rembg
    from PIL import Image

    with Image.open(input_path) as img:
        result = rembg.remove(img.convert("RGB"), session=session)
    tmp = output_path + ".tmp"
    result.save(tmp, "PNG", compress_level=1)  # fast; the fuse deletes it right away
    os.replace(tmp, output_path)


# --------------------------------------------------------------------------
# Shared worker
# --------------------------------------------------------------------------


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def acquire_lock(path):
    """Exclusive non-blocking lock; returns the open file or None if taken."""
    f = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def heartbeat(path, stop):
    while not stop.is_set():
        try:
            write_atomic(path, f"{int(time.time())} {os.getpid()}")
        except OSError:
            pass  # the fuse may be reading it (Windows); try again next second
        stop.wait(1.0)


def remove_stale_files(spool, max_age=3600):
    now = time.time()
    for entry in os.scandir(spool):
        if entry.name.startswith("worker."):
            continue
        try:
            if now - entry.stat().st_mtime > max_age:
                os.remove(entry.path)
        except OSError:
            pass


def handle_request(req_path, sessions):
    reply = req_path[: -len(".req")]
    try:
        with open(req_path, encoding="utf-8") as f:
            input_path, output_path, model = [line.strip() for line in f][:3]
    except Exception as e:
        write_atomic(reply + ".err", f"Bad request: {e}")
        return
    finally:
        try:
            os.remove(req_path)
        except OSError:
            pass

    try:
        if model in sessions:
            sessions.move_to_end(model)
        else:
            sessions[model] = load_session(model)
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        session, provider = sessions[model]

        started = time.time()
        remove_background(session, input_path, output_path)
        write_atomic(reply + ".done", f"ok {provider} {time.time() - started:.3f}")
    except Exception as e:
        log(f"[ERROR] {os.path.basename(req_path)}: {type(e).__name__}: {e}")
        write_atomic(reply + ".err", f"{type(e).__name__}: {e}")


def serve(spool):
    os.makedirs(spool, exist_ok=True)
    lock = acquire_lock(os.path.join(spool, "worker.lock"))
    if lock is None:
        return  # another worker is already running

    ensure_output(os.path.join(spool, "worker.log"))
    log(f"Worker started: pid {os.getpid()}, {sys.executable}")

    stop = threading.Event()
    beat_path = os.path.join(spool, "worker.heartbeat")
    threading.Thread(target=heartbeat, args=(beat_path, stop), daemon=True).start()

    script_mtime = os.path.getmtime(__file__)
    sessions = OrderedDict()
    last_request = time.time()
    remove_stale_files(spool)

    try:
        while True:
            requests = sorted(
                (e for e in os.scandir(spool) if e.name.endswith(".req")),
                key=lambda e: e.stat().st_mtime,
            )
            for entry in requests:
                handle_request(entry.path, sessions)
                last_request = time.time()

            if not requests:
                if time.time() - last_request > IDLE_TIMEOUT:
                    log("Idle timeout; exiting.")
                    break
                if os.path.getmtime(__file__) != script_mtime:
                    log("bg_remover.py changed; exiting so the next request uses it.")
                    break
                time.sleep(0.005)
    finally:
        stop.set()
        try:
            os.remove(beat_path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# One-shot command line
# --------------------------------------------------------------------------


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(sys.argv[2])
        return

    ensure_output()
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)

    input_path, output_path, model = sys.argv[1:4]
    if not os.path.exists(input_path):
        log(f"[ERROR] Input file does not exist: {input_path}")
        sys.exit(1)
    try:
        session, provider = load_session(model)
        started = time.time()
        remove_background(session, input_path, output_path)
        log(f"[COMPLETE] {output_path} ({provider}, {time.time() - started:.2f}s)")
    except Exception as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
