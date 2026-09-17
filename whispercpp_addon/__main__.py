"""whisper.cpp add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline speech-to-text with whisper.cpp through pywhispercpp: CPU only, no
torch, no cloud account. Serves `asr.transcribe`. It is Subtitld's former
built-in `whispercpp` provider moved out of process — same id, same model
files, same phrase regrouping — so existing users keep their settings and
their downloaded models.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:whispercpp]` and surfaces.

Models
------
GGML models download on first use from the whisper.cpp model repository on
Hugging Face as `ggml-<model>.bin`. The directory is resolved first hit wins:

  1. `options['models_dir']` / `$WHISPERCPP_MODELS_DIR` — the Configure field.
  2. `$SUBTITLD_MODELS_DIR/whispercpp` — exported by the host.
  3. `platformdirs.user_cache_dir('subtitld')/models/whispercpp` — exactly the
     directory the built-in provider used, so a model downloaded by an older
     Subtitld (up to 3.1 GB for large-v3) is picked up with no download.

Never inside the add-on's install directory: the host deletes that on every
upgrade. Downloads go to a `.part` sibling and are renamed into place only
once complete, so a partial file can never pass for a model.

Configuration
-------------
Resolved per request, first hit wins: `params['options']` → the
`WHISPERCPP_<KEY>` env var the host sets from the Configure dialog → default.
Values that are None or blank count as unset, matching the host, which never
exports blank values.

Cancellation
------------
A `cancel` frame sets the request's Event. It is read in three places: the
download loop, every stage boundary, and whisper.cpp's `abort_callback`, which
whisper.cpp polls during decoding — an in-flight transcription stops in well
under a second. whisper.cpp returns *normally* after an abort, with whatever
it decoded so far, so the flag is checked again once `transcribe()` returns:
a cancelled request always ends in error `cancelled`, never a truncated
`result`.

Model cache
-----------
One whisper context stays loaded for the life of the process, keyed by model
path. Subtitld's Record mode calls `transcribe` once per spoken phrase, and
the built-in reloaded the model every time (seconds per phrase on large-v3).
Switching models frees the old context first, so two are never resident. One
lock serialises all use of the context: whisper.cpp is not re-entrant on it.

Native pitfalls handled here
----------------------------
* A failed model load is invisible to Python. pywhispercpp wraps the NULL
  context in a perfectly ordinary-looking object, and the first decode then
  segfaults, taking the process with it. So a `whisper_log_set` callback
  records whisper.cpp's log while a model loads, and the load only counts if
  whisper.cpp reached its last initialisation step (`compute buffer
  (decode)`). "No ERROR line" would not do: a GPU backend that fails to start
  logs an ERROR, and whisper.cpp then carries on, correctly, on the CPU. A
  failed load becomes a `model_missing` (or `oom`) error that names the file.
  The same callback routes whisper.cpp's ~35 init lines to this log at DEBUG
  level instead of flooding the host's stderr.
* Given zero samples, whisper.cpp returns early without clearing its
  previous results, and pywhispercpp hands back the PREVIOUS request's text.
  Audio under 0.1 s never reaches it; whisper.cpp refuses that anyway.
* pywhispercpp's `Model.__del__` raises at interpreter teardown, so the model
  is freed explicitly before `main()` returns.

stdio
-----
The protocol owns the real stdin/stdout. `_claim_protocol_stdio` moves them
to private descriptors, then points fd 0 at devnull and fd 1 at stderr. So
nothing else can read a host frame or write into the frame stream: not a
stray `print()`, not native `printf` inside whisper.cpp, and not an ffmpeg
child (which reads keystrokes from an inherited stdin).
"""

from __future__ import annotations

import errno
import json
import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import wave
from pathlib import Path

# numpy's bundled OpenBLAS starts one thread per core when it loads. This
# add-on only uses numpy to hold the audio buffer, so those threads would be
# pure overhead. Must happen before numpy is first imported. (Deliberately NOT
# OMP_NUM_THREADS: whisper.cpp's own OpenMP pool is sized by `threads`.)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# Deferred-failure import: a broken native runtime (missing shared library,
# wrong architecture) is reported to the host as a `hello_error` frame instead
# of dying with a traceback before the handshake.
try:
    import numpy as np
    from pywhispercpp.model import Model as _WhisperModel  # runs the delvewheel DLL patch first
    import _pywhispercpp as _pw
except Exception as _exc:  # noqa: BLE001 — ImportError, OSError from the loader, ...
    np = None
    _WhisperModel = None
    _pw = None
    _IMPORT_ERROR: Exception | None = _exc
else:
    _IMPORT_ERROR = None

ADDON_ID = 'whispercpp'
ADDON_VERSION = '0.0.1'
PROTOCOL_VERSION = 1

# Error codes. Mirrors `subtitld.modules.addons.protocol.ErrorCode`; inlined so
# the frozen binary takes no runtime dependency on the host package.
ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK_UNAVAILABLE = 'network_unavailable'

# Whisper's 99 ISO-639-1 codes (openai/whisper tokenizer.py), bare. The host's
# language matcher resolves `pt-br` against `pt`. `manifest.json` must list
# exactly these — tests/test_manifest.py enforces it.
_WHISPER_LANGUAGES = (
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
    'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi',
    'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
    'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
    'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
    'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
    'yi', 'yo', 'zh',
)
_LANGUAGE_SET = frozenset(_WHISPER_LANGUAGES)
# Tags a host may send for a language Whisper spells differently.
# Subtitld offers Chinese only as `cmn-hans-cn`, `cmn-hant-tw`, `yue-hant-hk`
# (Mandarin, Cantonese). Whisper's `zh` covers both; its separate `yue` token
# exists in large-v3 only, so every model gets `zh`.
_LANGUAGE_ALIASES = {
    'jv': 'jw', 'iw': 'he', 'in': 'id', 'ji': 'yi', 'fil': 'tl', 'nb': 'no',
    'cmn': 'zh', 'yue': 'zh',
}

# Model ids, URL, filename and User-Agent are byte-identical to the built-in
# provider. Any drift orphans models users already downloaded.
_MODEL_BASE_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'
_USER_AGENT = 'subtitld-whispercpp/1'
_DEFAULT_MODEL = 'base'
_AVAILABLE_MODELS = (
    'tiny', 'tiny.en',
    'base', 'base.en',
    'small', 'small.en',
    'medium', 'medium.en',
    'large-v3',
)
_DOWNLOAD_CHUNK_BYTES = 256 * 1024
_GGML_MAGIC = b'lmgg'  # 0x67676d6c little-endian, the first 4 bytes of every model
# A `.part` older than this belongs to a dead process (a live download keeps
# touching its file). Without the sweep, a cancelled large-v3 download under
# the built-in left 3 GB of garbage behind forever.
_STALE_PART_SEC = 6 * 3600
# Free space kept in reserve on top of the model's size before downloading.
_DISK_MARGIN_BYTES = 64 * 1024 * 1024

_SAMPLE_RATE = 16000
_MIN_SAMPLES = _SAMPLE_RATE // 10  # whisper.cpp refuses input under 100 ms

# The last line whisper_init_state() logs, after every allocation succeeded.
# A context without it is NULL (whisper.cpp 1.8.5, pinned via pywhispercpp).
_LOAD_OK_MARKER = 'compute buffer (decode)'

# ggml_log_level
_GGML_LOG_WARN = 3
_GGML_LOG_ERROR = 4
_GGML_LOG_CONT = 5

log = logging.getLogger('whispercpp_addon')
_native_log = logging.getLogger('whispercpp_addon.native')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`: whisper.cpp is native code
    and writes to fd 1 directly, and a child process inherits fds, not Python
    objects. Falls back to the Python-level swap if the fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_partial(req_id: str, segment: dict) -> None:
    _emit({'id': req_id, 'type': 'partial', 'data': {'segment': segment}})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class _Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _opt(options: dict, key: str, env_name: str, default):
    """Request option, then env, then default. None and blank strings are unset."""
    value = options.get(key) if isinstance(options, dict) else None
    if isinstance(value, str):
        value = value.strip()
    if value is not None and value != '':
        return value
    raw = _env(env_name)
    return raw if raw else default


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() not in ('0', 'false', 'no', 'off', '')


def _opt_bool(options: dict, key: str, env_name: str, default: bool) -> bool:
    """Three-way resolve for a boolean.

    `options.get(key) or default` would be wrong — an explicit False is falsy
    and would silently become the default. The host exports booleans as
    '1'/'0'; a hand-edited config may hold the string "false".
    """
    value = options.get(key) if isinstance(options, dict) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.strip():
        return _parse_bool(value)
    raw = _env(env_name)
    if raw:
        return _parse_bool(raw)
    return default


def _resolve_model(options: dict) -> str:
    model = str(_opt(options, 'model', 'WHISPERCPP_MODEL', _DEFAULT_MODEL))
    if model not in _AVAILABLE_MODELS:
        log.warning('unknown model %r, falling back to %s', model, _DEFAULT_MODEL)
        model = _DEFAULT_MODEL
    return model


def _resolve_threads(options: dict) -> int:
    """0 = let whisper.cpp decide. The Configure dialog stores a string."""
    raw = _opt(options, 'threads', 'WHISPERCPP_THREADS', 0)
    try:
        threads = int(str(raw).strip() or 0)
    except (TypeError, ValueError):
        log.warning('ignoring non-numeric threads value %r', raw)
        threads = 0
    return max(0, min(threads, 256))


def _resolve_language(raw) -> str:
    """Whisper language code, or '' for auto-detect.

    The host sends BCP-47-ish tags (`pt-br`, `en_US`); Whisper wants the bare
    primary subtag. An unknown code is an error rather than a guess: the
    built-in sliced `[:2]`, which turned `haw` into Hausa and `fil` into
    Finnish, and whisper.cpp itself only logs an unknown code.
    """
    tag = str(raw or '').strip().lower().replace('_', '-')
    if not tag or tag in ('auto', 'und', 'mul'):
        return ''
    primary = tag.split('-', 1)[0]
    primary = _LANGUAGE_ALIASES.get(primary, primary)
    if primary in _LANGUAGE_SET:
        return primary
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'Whisper does not support the language {raw!r}.')


def _models_dir(options: dict | None = None) -> Path:
    """Where `ggml-<model>.bin` lives. See the module docstring for the order."""
    raw = _opt(options or {}, 'models_dir', 'WHISPERCPP_MODELS_DIR', '')
    if raw:
        path = Path(os.path.expanduser(str(raw)))
        # The Configure dialog's `file` field opens a FILE picker. Accept a
        # file inside the folder (typically one of the models) as the folder.
        if path.is_file():
            path = path.parent
        return path
    base = _env('SUBTITLD_MODELS_DIR')
    if base:
        return Path(os.path.expanduser(base)) / 'whispercpp'
    # Same call the host makes for `session.PATH_SUBTITLD_DATA_MODELS`
    # (`platformdirs.user_cache_dir('subtitld') / 'models'`).
    import platformdirs
    return Path(platformdirs.user_cache_dir('subtitld')) / 'models' / 'whispercpp'


def _model_base_url() -> str:
    # Mirror/offline-test override. Not a Configure field on purpose.
    return (_env('WHISPERCPP_ADDON_MODEL_BASE_URL') or _MODEL_BASE_URL).rstrip('/')


def _temp_root() -> str:
    raw = _env('SUBTITLD_TEMP_DIR')
    if raw and os.path.isdir(raw):
        return raw
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _convert_to_wav(ffmpeg: str, src: str, dst: Path, cancel: threading.Event) -> None:
    """Decode anything ffmpeg reads into 16 kHz mono s16 WAV.

    We never let pywhispercpp touch the source file: its loader hardcodes
    `shutil.which('ffmpeg')` and rejects any WAV that is not already 16 kHz.
    `-nostdin` plus stdin=DEVNULL: ffmpeg polls stdin for keystrokes.
    """
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
           '-i', src, '-vn', '-sn', '-dn',
           '-ac', '1', '-ar', str(_SAMPLE_RATE), '-c:a', 'pcm_s16le', '-f', 'wav',
           str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, startupinfo=_no_window_startupinfo())
    stderr = b''
    while True:
        try:
            _out, stderr = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set():
                proc.kill()
                proc.communicate()
                raise _Cancelled() from None
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        tail = ' / '.join((stderr or b'').decode('utf-8', 'replace').strip().splitlines()[-5:])
        if 'No space left' in tail:
            raise _AddonError(ERR_DISK_FULL, f'Out of disk space while decoding the audio: {tail}')
        raise _AddonError(ERR_INTERNAL,
                          f'Could not decode the audio with ffmpeg '
                          f'(exit {proc.returncode}): {tail or "no output"}')


def _read_wav(path: Path):
    """(float32 samples in [-1, 1), duration in seconds)."""
    with wave.open(str(path), 'rb') as w:
        channels, width, rate, frames = (w.getnchannels(), w.getsampwidth(),
                                         w.getframerate(), w.getnframes())
        if (channels, width, rate) != (1, 2, _SAMPLE_RATE):
            raise _AddonError(ERR_INTERNAL,
                              f'ffmpeg produced {channels} ch / {width * 8} bit / {rate} Hz, '
                              f'expected mono 16-bit {_SAMPLE_RATE} Hz')
        raw = w.readframes(frames)
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    return samples, len(samples) / float(_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
def _model_filename(model: str) -> str:
    return f'ggml-{model}.bin'


def _has_ggml_magic(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(4) == _GGML_MAGIC
    except OSError:
        return False


def _sweep_stale_parts(target: Path) -> None:
    now = time.time()
    for part in target.parent.glob(target.name + '*.part'):
        try:
            if now - part.stat().st_mtime > _STALE_PART_SEC:
                part.unlink()
                log.info('removed stale partial download %s', part)
        except OSError:
            pass


_download_lock = threading.Lock()


def _ensure_model_file(model: str, models_dir: Path, state: '_Request',
                       lo: float, hi: float) -> tuple[Path, bool]:
    """(path, downloaded). Downloads once, atomically, with byte progress."""
    target = models_dir / _model_filename(model)
    if target.is_file() and target.stat().st_size > 0:
        if not _has_ggml_magic(target):
            raise _AddonError(ERR_MODEL_MISSING,
                              f'{target} is not a whisper.cpp model file. '
                              f'Delete it and try again to download a fresh copy.')
        return target, False

    with _download_lock:
        # Another request in this process may have just finished it.
        if target.is_file() and target.stat().st_size > 0:
            return target, False
        _download(model, target, state, lo, hi)
        return target, True


def _download(model: str, target: Path, state: '_Request', lo: float, hi: float) -> None:
    req_id = state.req_id
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _classify_os_error(exc, f'Cannot create the models folder {target.parent}') from exc
    _sweep_stale_parts(target)
    # Per-process name: the Record panel runs a second add-on process, and two
    # processes must never interleave writes into one file.
    part = target.with_name(f'{target.name}.{os.getpid()}.part')
    url = f'{_model_base_url()}/{_model_filename(model)}'
    log.info('downloading %s -> %s', url, target)
    _emit_progress(req_id, lo, f'Downloading {model} model (first use)...')
    request = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            try:
                total = int(resp.headers.get('Content-Length') or 0)
            except (TypeError, ValueError):
                total = 0
            if total:
                free = shutil.disk_usage(target.parent).free
                if free < total + _DISK_MARGIN_BYTES:
                    raise _AddonError(
                        ERR_DISK_FULL,
                        f'Not enough disk space for the {model} model: it needs '
                        f'{total // 2**20} MB and {target.parent} has {free // 2**20} MB free.')
            written = 0
            head = b''
            last_step = -1
            total_mb = total // 2**20
            with open(part, 'wb') as out:
                while True:
                    if state.cancel.is_set():
                        raise _Cancelled()
                    chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    if len(head) < 4:
                        head += chunk[:4 - len(head)]
                        if len(head) == 4 and head != _GGML_MAGIC:
                            raise _AddonError(
                                ERR_MODEL_MISSING,
                                f'{url} did not return a whisper.cpp model '
                                f'(got {head!r}). The download mirror may be misbehaving.',
                                retryable=True)
                    out.write(chunk)
                    written += len(chunk)
                    if total:
                        step = written * 200 // total  # at most 200 progress frames
                        if step != last_step:
                            last_step = step
                            _emit_progress(
                                req_id, lo + (hi - lo) * (written / total),
                                f'Downloading {model} model... '
                                f'{written // 2**20} / {total_mb} MB')
                out.flush()
                os.fsync(out.fileno())
            if total and written != total:
                raise _AddonError(ERR_NETWORK_UNAVAILABLE,
                                  f'Model download ended early ({written} of {total} bytes).',
                                  retryable=True)
            if written < 4 or head != _GGML_MAGIC:
                raise _AddonError(ERR_MODEL_MISSING,
                                  f'{url} returned no model data.', retryable=True)
        # Atomic: a partial file can never masquerade as a finished one.
        os.replace(part, target)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    _emit_progress(req_id, hi, f'Downloaded {model} model')
    log.info('downloaded %s (%d bytes)', target, target.stat().st_size)


# ---------------------------------------------------------------------------
# whisper.cpp engine: native log routing + the cached context
# ---------------------------------------------------------------------------
# While a model loads or a decode runs (always under `_engine_lock`),
# whisper.cpp's log lines are collected here as (level, text).
_native_capture: list[tuple[int, str]] | None = None
_native_last_level = 0


def _on_native_log(level, text, *_rest) -> None:
    # Called from native code, possibly on a whisper.cpp thread. It must never
    # raise: an exception here unwinds through C.
    global _native_last_level
    try:
        level = int(level)
        if level == _GGML_LOG_CONT:
            level = _native_last_level
        else:
            _native_last_level = level
        msg = str(text or '').rstrip()
        if not msg:
            return
        sink = _native_capture
        if sink is not None:
            sink.append((level, msg))
        if level >= _GGML_LOG_ERROR:
            _native_log.warning('%s', msg)
        elif level == _GGML_LOG_WARN:
            _native_log.info('%s', msg)
        else:
            _native_log.debug('%s', msg)
    except BaseException:  # noqa: BLE001
        pass


def _install_native_log() -> bool:
    if _pw is None or not hasattr(_pw, 'whisper_log_set'):
        return False
    try:
        _pw.whisper_log_set(_on_native_log)
        return True
    except Exception:  # noqa: BLE001
        log.warning('could not install the whisper.cpp log callback', exc_info=True)
        return False


_native_log_installed = False
_engine_lock = threading.Lock()
_engine: dict = {'path': None, 'model': None, 'default_threads': 4}


def _free_model(model) -> None:
    ctx = getattr(model, '_ctx', None)
    if ctx is None:
        return
    try:
        _pw.whisper_free(ctx)  # whisper_free(NULL) is a no-op
    finally:
        # Makes pywhispercpp's __del__ a no-op: it raises at interpreter
        # teardown, and must never free the context a second time.
        model._ctx = None


def _drop_model_locked() -> None:
    model = _engine['model']
    _engine['path'] = None
    _engine['model'] = None
    if model is not None:
        log.info('freeing whisper model')
        _free_model(model)


def _load_failure(lines: list[tuple[int, str]], path: Path) -> _AddonError | None:
    """whisper.cpp's own verdict on a model load, read from its log."""
    if any(_LOAD_OK_MARKER in text for _level, text in lines):
        return None
    errors = [text for level, text in lines if level >= _GGML_LOG_ERROR]
    detail = errors[-1] if errors else 'initialisation did not complete'
    if any('alloc' in e.lower() or 'memory' in e.lower() for e in errors):
        return _AddonError(ERR_OOM, f'Not enough memory to load {path.name}: {detail}')
    return _AddonError(
        ERR_MODEL_MISSING,
        f'whisper.cpp could not load {path} ({detail}). The file is probably '
        f'corrupt or incomplete: delete it and try again to download a fresh copy.')


def _load_model_locked(path: Path, on_load):
    """The cached context for `path`, loading it (and freeing any other) if needed."""
    global _native_capture
    key = str(path)
    if _engine['path'] == key and _engine['model'] is not None:
        return _engine['model']
    _drop_model_locked()
    on_load()
    if not path.is_file():  # pywhispercpp would go and download it itself
        raise _AddonError(ERR_MODEL_MISSING, f'Model file disappeared: {path}')
    kwargs = {'print_progress': False}
    if _native_log_installed:
        kwargs['redirect_whispercpp_logs_to'] = False  # the log callback already routes it
    else:
        kwargs['redirect_whispercpp_logs_to'] = None   # devnull. NEVER sys.stdout.
    started = time.monotonic()
    _native_capture = []
    try:
        model = _WhisperModel(key, **kwargs)
    finally:
        lines, _native_capture = _native_capture, None
    # Without the log callback there is no verdict to read; the magic-byte
    # check in _ensure_model_file is then the only guard.
    failure = _load_failure(lines, path) if _native_log_installed else None
    if failure is None and getattr(model, '_ctx', None) is None:
        failure = _AddonError(ERR_MODEL_MISSING, f'whisper.cpp returned no context for {path}')
    if failure is not None:
        _free_model(model)
        raise failure
    try:
        default_threads = int(model._params.n_threads)
    except Exception:  # noqa: BLE001
        default_threads = 4
    _engine.update(path=key, model=model, default_threads=max(1, default_threads))
    log.info('loaded %s in %.2fs (default threads %d)',
             path.name, time.monotonic() - started, _engine['default_threads'])
    return model


def _segment_to_dict(seg) -> dict | None:
    # pywhispercpp >= 1.2 exposes `t0`/`t1` in centiseconds; older builds gave
    # `.start`/`.end` in seconds. Accept both so a wheel bump can't silently
    # zero every timestamp.
    t0 = getattr(seg, 't0', None)
    t1 = getattr(seg, 't1', None)
    if t0 is None and t1 is None:
        start = float(getattr(seg, 'start', 0.0) or 0.0)
        end = float(getattr(seg, 'end', 0.0) or 0.0)
    else:
        start = float(t0 or 0) / 100.0
        end = float(t1 or 0) / 100.0
    text = (getattr(seg, 'text', '') or '').strip()
    if not text:
        return None
    return {'start': start, 'end': end, 'text': text, 'speaker': 'A'}


def _run_whisper(model, samples, duration: float, language: str, threads: int,
                 state: '_Request', lo: float) -> list[dict]:
    """Decode under `_engine_lock`. Streams each finalized segment as a partial."""
    global _native_capture
    req_id = state.req_id
    cancel = state.cancel
    collected: list[dict] = []
    span = 0.999 - lo  # the final `result` is what snaps the bar to 100%

    def on_segment(seg) -> None:
        try:
            item = _segment_to_dict(seg)
            if item is None:
                return
            collected.append(item)
            _emit_partial(req_id, item)
            if duration > 0.0:
                fraction = min(1.0, max(0.0, item['end'] / duration))
                _emit_progress(req_id, lo + span * fraction,
                               f'Transcribing... {int(round(item["end"]))}/{int(round(duration))}s')
        except Exception:  # noqa: BLE001 — never raise into native code
            log.exception('segment callback failed')

    def should_abort() -> bool:
        # Polled by whisper.cpp during decoding, under the GIL: keep it trivial.
        return cancel.is_set()

    _native_capture = []
    try:
        model.transcribe(
            samples,
            new_segment_callback=on_segment,
            abort_callback=should_abort,
            # Params persist on the cached model, so both are passed EVERY
            # call. '' = auto-detect; pywhispercpp's own default is 'en'.
            language=language,
            n_threads=threads or _engine['default_threads'],
        )
    finally:
        lines, _native_capture = _native_capture, None
        try:  # drop references to this request's closures
            _pw.assign_new_segment_callback(model._params, None)
            _pw.assign_abort_callback(model._params, None)
            model._new_segment_callback = None
        except Exception:  # noqa: BLE001
            pass

    if cancel.is_set():
        raise _Cancelled()
    errors = [text for level, text in lines if level >= _GGML_LOG_ERROR]
    if errors and not collected:
        raise _AddonError(ERR_INTERNAL, f'whisper.cpp failed: {errors[-1]}')
    # Only the callback's list is trusted. On an early-return path whisper.cpp
    # leaves its previous results in place, and the return value would hand
    # back the LAST request's text; the callback only ever sees fresh output.
    return collected


def _acquire_engine(state: '_Request', progress_value: float) -> None:
    waiting = False
    while not _engine_lock.acquire(timeout=0.25):
        if state.cancel.is_set() or _shutting_down.is_set():
            raise _Cancelled()
        if not waiting:
            waiting = True
            _emit_progress(state.req_id, progress_value,
                           'Waiting for the previous transcription to finish...')


# ---------------------------------------------------------------------------
# Phrase regrouping — verbatim from the built-in provider (pure stdlib).
# ---------------------------------------------------------------------------
#
# whisper.cpp emits segments at its own decoding cadence — typically a
# few-second window or a detected long pause — which means a single
# sentence often gets sliced across two adjacent segments (and, less
# commonly, several short sentences land in one). For subtitle work the
# user wants one row per phrase, so we post-process: atomize at
# sentence terminators (with time spans distributed by character-length
# ratio inside a split), then re-merge consecutive atoms until each row
# ends with a terminator or hits the readability cap. Speaker changes
# always force a break.
#
# The character-ratio interpolation is a coarse approximation — real
# speech doesn't have a uniform char/sec rate — but it's accurate enough
# for a manual-edit first pass and avoids the cost of asking whisper.cpp
# for word-level timestamps (which roughly doubles inference time).

# Sentence-ending punctuation across scripts. ASCII terminators,
# full-width CJK forms, Devanagari danda/double-danda, and the Arabic
# question mark. We intentionally OMIT the opening Spanish `¿`/`¡` since
# those don't end a phrase.
_PHRASE_TERMINATORS = frozenset('.?!。？！．।॥؟')

# Characters that may trail a terminator and still belong to the same
# sub-phrase — closing quotes, brackets, etc. So `"hello."` stays
# together when we split.
_PHRASE_TRAILING_CLOSERS = frozenset('"\')]}»›\u201d\u2019')

# Subtitle readability caps used as a fallback when whisper omits
# punctuation (common for low-quality audio or certain languages).
# Tuned to typical captioning guidelines: ~2 lines of ~42 chars at the
# canonical ~17 cps reading rate.
_PHRASE_MAX_CHARS = 84
_PHRASE_MAX_DURATION_SEC = 7.0


def _ends_with_terminator(text: str) -> bool:
    """True if ``text`` ends with sentence-final punctuation.

    Strips up to a handful of trailing closing-quotes/brackets first so
    `She said "hello."` still registers as a terminated sentence.
    """
    t = text.rstrip()
    for _ in range(4):
        if not t:
            return False
        if t[-1] in _PHRASE_TRAILING_CLOSERS:
            t = t[:-1].rstrip()
            continue
        break
    return bool(t) and t[-1] in _PHRASE_TERMINATORS


def _split_segment_on_terminators(seg: dict) -> list[dict]:
    """Split one raw whisper segment at internal sentence terminators.

    The original time span is distributed across sub-segments by
    character-length ratio. Each terminator (with any trailing closing
    quotes/brackets) closes the current sub-phrase; the remainder
    becomes the next.
    """
    text = (seg.get('text') or '').strip()
    if not text:
        return []
    start = float(seg.get('start', 0.0))
    end = float(seg.get('end', 0.0))
    speaker = seg.get('speaker', 'A')
    duration = max(0.0, end - start)
    total_chars = max(1, len(text))

    # Find every cut point — index just past a terminator (and any
    # trailing closing quotes/brackets).
    cut_points: list[int] = []
    i = 0
    while i < len(text):
        if text[i] in _PHRASE_TERMINATORS:
            j = i + 1
            while j < len(text) and text[j] in _PHRASE_TRAILING_CLOSERS:
                j += 1
            cut_points.append(j)
            i = j
        else:
            i += 1
    if not cut_points or cut_points[-1] < len(text):
        cut_points.append(len(text))

    subs: list[dict] = []
    prev = 0
    for cut in cut_points:
        chunk = text[prev:cut].strip()
        if chunk:
            subs.append({
                'start': start + duration * (prev / total_chars),
                'end': start + duration * (cut / total_chars),
                'text': chunk,
                'speaker': speaker,
            })
        prev = cut
    return subs


def _regroup_into_phrases(segments: list[dict]) -> list[dict]:
    """Rewrite raw whisper segments into phrase-aligned subtitle rows.

    Two passes: atomize each segment at internal sentence terminators,
    then re-merge consecutive atoms until each phrase either ends with
    a terminator or exceeds the readability caps. Speaker changes force
    a phrase boundary regardless.

    Returns a new list; does not mutate the input. Preserves order.
    """
    if not segments:
        return []

    atoms: list[dict] = []
    for seg in segments:
        atoms.extend(_split_segment_on_terminators(seg))
    if not atoms:
        return []

    phrases: list[dict] = []
    current: dict | None = None
    for atom in atoms:
        if current is None:
            current = dict(atom)
            continue
        # Never merge across speakers — even with no punctuation, a
        # diarization change is a hard phrase boundary.
        if atom['speaker'] != current['speaker']:
            phrases.append(current)
            current = dict(atom)
            continue
        # Terminator boundaries are sacred: if the current phrase
        # already ends with sentence-final punctuation, don't merge
        # the next atom into it. This is what produces one-row-per-
        # sentence — even short sentences stay split.
        if _ends_with_terminator(current['text']):
            phrases.append(current)
            current = dict(atom)
            continue

        # current is mid-sentence (no terminator) — try to absorb the
        # next atom to complete it.
        merged_text = current['text'] + ' ' + atom['text']
        merged_duration = atom['end'] - current['start']
        would_overflow = (
            len(merged_text) > _PHRASE_MAX_CHARS
            or merged_duration > _PHRASE_MAX_DURATION_SEC
        )
        if would_overflow:
            # No punctuation to lean on and we're past the readability
            # cap. Force-close current as-is — a 7-second row that ends
            # mid-thought still beats cramming the next chunk into a
            # runaway 30-second one.
            phrases.append(current)
            current = dict(atom)
            continue

        current['text'] = merged_text
        current['end'] = atom['end']
        # No early-close-on-terminator here: the top-of-loop check on
        # the NEXT iteration handles it. If this was the last atom we
        # fall through to the trailing append below.

    if current is not None:
        phrases.append(current)
    return phrases


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active_lock = threading.Lock()
_active: dict[str, _Request] = {}
_shutting_down = threading.Event()


def _classify_os_error(exc: OSError, context: str) -> _AddonError:
    if getattr(exc, 'errno', None) in (errno.ENOSPC, getattr(errno, 'EDQUOT', -1)):
        return _AddonError(ERR_DISK_FULL, f'{context}: out of disk space ({exc})')
    return _AddonError(ERR_INTERNAL, f'{context}: {exc}')


def _classify(exc: BaseException) -> tuple[str, str, bool]:
    """Map an unexpected exception onto a protocol error."""
    if isinstance(exc, _AddonError):
        return exc.code, exc.message, exc.retryable
    if isinstance(exc, MemoryError):
        return ERR_OOM, 'Out of memory. Try a smaller model.', False
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403, 404, 410):
            return (ERR_MODEL_MISSING,
                    f'The model could not be downloaded (HTTP {exc.code}).', False)
        return (ERR_NETWORK_UNAVAILABLE,
                f'Model download failed (HTTP {exc.code}). Check your internet '
                f'connection and try again.', True)
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        reason = getattr(exc, 'reason', exc)
        return (ERR_NETWORK_UNAVAILABLE,
                f'Model download failed ({reason}). Check your internet connection '
                f'and try again. Once a model is downloaded, transcription is fully offline.',
                True)
    if isinstance(exc, OSError):
        err = _classify_os_error(exc, 'I/O error')
        return err.code, err.message, err.retryable
    return ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False


def _handle_transcribe(state: _Request, params: dict) -> None:
    req_id = state.req_id
    options = params.get('options')
    if not isinstance(options, dict):
        options = {}
    audio_path = params.get('audio_path')
    if not isinstance(audio_path, str) or not audio_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'audio_path is required')
        return
    if not os.path.isfile(audio_path):
        _emit_error(req_id, ERR_BAD_PARAMS, f'audio_path is not a file: {audio_path!r}')
        return
    if _IMPORT_ERROR is not None:
        _emit_error(req_id, ERR_INTERNAL, f'whisper.cpp runtime unavailable: {_IMPORT_ERROR}')
        return
    try:
        language = _resolve_language(params.get('language'))
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    model_id = _resolve_model(options)
    threads = _resolve_threads(options)
    phrase_grouping = _opt_bool(options, 'phrase_grouping', 'WHISPERCPP_PHRASE_GROUPING', True)
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    'ffmpeg not found. Set SUBTITLD_FFMPEG_EXECUTABLE or put ffmpeg on PATH.')
        return

    # The terminal frame is sent only after the temporary files are gone, so
    # a host that sees `result`/`error` never finds our WAV still on disk.
    workdir = None
    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        models_dir = _models_dir(options)
        log.info('transcribe %s: model=%s language=%s threads=%s grouping=%s models_dir=%s',
                 req_id, model_id, language or 'auto', threads or 'auto',
                 phrase_grouping, models_dir)
        if state.cancel.is_set():
            raise _Cancelled()
        try:
            workdir = Path(tempfile.mkdtemp(prefix='subtitld-whispercpp-', dir=_temp_root()))
        except OSError as exc:
            raise _classify_os_error(exc, 'Cannot create a temporary folder') from exc
        wav_path = workdir / 'audio.16k.wav'

        # 1. Decode: 0.00 -> 0.05
        _emit_progress(req_id, 0.0, 'Decoding audio...')
        _convert_to_wav(ffmpeg, audio_path, wav_path, state.cancel)
        samples, duration = _read_wav(wav_path)
        wav_path.unlink()
        if state.cancel.is_set():
            raise _Cancelled()
        _emit_progress(req_id, 0.05, 'Decoded audio')

        if samples.size < _MIN_SAMPLES:
            log.info('transcribe %s: %.3fs of audio, nothing to transcribe', req_id, duration)
            segments: list[dict] = []
        else:
            # 2. Model file: 0.05 -> 0.90, only when it has to be downloaded.
            model_path, downloaded = _ensure_model_file(model_id, models_dir, state, 0.05, 0.90)
            if state.cancel.is_set():
                raise _Cancelled()
            load_at = 0.90 if downloaded else 0.05
            transcribe_at = 0.90 if downloaded else 0.60

            # 3. Load (-> 0.60) and transcribe (-> 1.00), holding the engine.
            _acquire_engine(state, load_at)
            try:
                if state.cancel.is_set():
                    raise _Cancelled()
                model = _load_model_locked(
                    model_path,
                    on_load=lambda: _emit_progress(req_id, load_at, f'Loading {model_id} model...'))
                if state.cancel.is_set():
                    raise _Cancelled()
                _emit_progress(req_id, transcribe_at, 'Transcribing...')
                started = time.monotonic()
                segments = _run_whisper(model, samples, duration, language, threads,
                                        state, transcribe_at)
                log.info('transcribe %s: %d segment(s) from %.1fs of audio in %.2fs',
                         req_id, len(segments), duration, time.monotonic() - started)
            finally:
                _engine_lock.release()

        if state.cancel.is_set():
            raise _Cancelled()
        if phrase_grouping:
            segments = _regroup_into_phrases(segments)
        outcome = ('result', {
            'segments': segments,
            'duration_sec': round(duration, 3),
            'model': model_id,
        })
    except _Cancelled:
        log.info('transcribe %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        code, message, retryable = _classify(exc)
        if code == ERR_INTERNAL and not isinstance(exc, _AddonError):
            log.exception('transcribe %s failed', req_id)  # unexpected: keep the traceback
        else:
            log.warning('transcribe %s failed: [%s] %s', req_id, code, message)
        outcome = ('error', code, message, retryable)
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'whispercpp-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'asr.transcribe',
            'languages': list(_WHISPER_LANGUAGES),
            # Segments stream as `partial` frames while decoding, but this is
            # batch transcription, not the live `asr.stream` task.
            'streaming': False,
            'voice_clone': False,
        }],
    })


def _send_hello_error(message: str, code: str = ERR_INTERNAL) -> None:
    _emit({'type': 'hello_error', 'code': code, 'message': message})


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('\ufeff')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'asr.transcribe':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('asr.transcribe without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_transcribe, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    _shutting_down.set()
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _release_native(workers_stopped: bool) -> bool:
    """Free the model and detach the log callback. False if the engine is busy."""
    if _pw is None:
        return True
    freed = _engine_lock.acquire(timeout=1.0 if workers_stopped else 0.0)
    if freed:
        try:
            _drop_model_locked()
        except Exception:  # noqa: BLE001
            log.exception('failed to free the whisper model')
        finally:
            _engine_lock.release()
    if freed and _native_log_installed:
        try:
            _pw.whisper_log_set(None)  # nothing may call back into Python after this
        except Exception:  # noqa: BLE001
            pass
    return freed


def main() -> int:
    global _native_log_installed
    # No-op unless frozen, where it keeps a multiprocessing helper that
    # re-launches this executable from running a second protocol loop.
    multiprocessing.freeze_support()
    _claim_protocol_stdio()
    logging.basicConfig(
        level=(_env('WHISPERCPP_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    try:
        from importlib.metadata import version
        pw_version = version('pywhispercpp')
    except Exception:  # noqa: BLE001
        pw_version = 'unknown'
    log.info('whispercpp-addon %s (pywhispercpp %s) on %s %s, python %s',
             ADDON_VERSION, pw_version, platform.system(), platform.machine(),
             platform.python_version())

    if _IMPORT_ERROR is not None:
        log.error('whisper.cpp runtime failed to load: %r', _IMPORT_ERROR)
        _send_hello_error(f'whisper.cpp runtime failed to load: {_IMPORT_ERROR}')
        return 1
    _native_log_installed = _install_native_log()
    if _find_ffmpeg() is None:
        # Not fatal: better to appear in the engine list and fail the first
        # request with a clear message than to vanish from the UI.
        log.warning('ffmpeg not found; requests will fail until one is available')

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
        freed = _release_native(stopped)
    if not freed:
        # A worker is still inside whisper.cpp (a multi-GB model load cannot
        # be interrupted). Normal teardown would free native state under it;
        # leave without teardown instead.
        log.warning('a transcription is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
