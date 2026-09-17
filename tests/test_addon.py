"""Tests for the whisper.cpp add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Layers:
  * **Offline** — option/language/model-dir resolution, phrase regrouping,
    error mapping. In-process; no ffmpeg, no model, no network.
  * **Protocol** — drives the add-on as a real subprocess over stdio.
  * **ffmpeg** — decoding, the download path against a local HTTP server, and
    load-failure handling with a fake model. Skipped without ffmpeg, which is
    why CI installs it.
  * **Transcription** — real whisper.cpp inference with `ggml-tiny.bin`
    (75 MB, downloaded once by the add-on itself). Opt-in with
    WHISPERCPP_TEST_TRANSCRIPTION=1. The model is kept in
    WHISPERCPP_TEST_MODELS_DIR (default: <tmp>/whispercpp-addon-test-models)
    — never in Subtitld's own model cache.

Set WHISPERCPP_TEST_BINARY to an ABSOLUTE path to run every subprocess test
against the frozen PyInstaller build instead of `python -m whispercpp_addon`.
CI does; it is the only thing that exercises native-library collection.

tests/data/jfk.flac is whisper.cpp's samples/jfk.wav (tag v1.8.5), FLAC-encoded:
11 s of President Kennedy's 1961 inaugural address, a US federal government
work in the public domain.
"""

from __future__ import annotations

import errno
import http.server
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from whispercpp_addon import __main__ as addon  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
JFK = REPO_ROOT / 'tests' / 'data' / 'jfk.flac'
BINARY = os.environ.get('WHISPERCPP_TEST_BINARY') or None
FFMPEG = shutil.which('ffmpeg')

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg not on PATH')
transcription = pytest.mark.skipif(
    FFMPEG is None or os.environ.get('WHISPERCPP_TEST_TRANSCRIPTION') != '1',
    reason='set WHISPERCPP_TEST_TRANSCRIPTION=1 (downloads ggml-tiny.bin, 75 MB) and install ffmpeg')

# Env vars from the developer's shell must not leak into the subprocess.
_SCRUB = ('WHISPERCPP_MODEL', 'WHISPERCPP_THREADS', 'WHISPERCPP_PHRASE_GROUPING',
          'WHISPERCPP_MODELS_DIR', 'WHISPERCPP_ADDON_MODEL_BASE_URL',
          'SUBTITLD_MODELS_DIR', 'SUBTITLD_TEMP_DIR')


# ---------------------------------------------------------------------------
# Subprocess harness
# ---------------------------------------------------------------------------
class _Addon:
    def __init__(self, env_extra=None, exe=BINARY):
        env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
        env.update(env_extra or {})
        env['PYTHONPATH'] = str(REPO_ROOT)
        env['PYTHONUNBUFFERED'] = '1'
        cmd = [exe] if exe else [sys.executable, '-m', 'whispercpp_addon']
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=str(REPO_ROOT))
        self._frames: queue.Queue = queue.Queue()
        self._stderr: list[bytes] = []
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        # Drained continuously: a chatty add-on must never block on a full pipe.
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self):
        for line in iter(self.proc.stdout.readline, b''):
            self._frames.put(line)
        self._frames.put(None)

    def _pump_stderr(self):
        for line in iter(self.proc.stderr.readline, b''):
            self._stderr.append(line)

    @property
    def stderr(self) -> str:
        return b''.join(self._stderr).decode('utf-8', 'replace')

    def send(self, frame):
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self, timeout=60.0) -> dict:
        try:
            line = self._frames.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError(f'no frame within {timeout}s\n--- stderr ---\n{self.stderr}') from None
        if line is None:
            raise AssertionError(f'add-on closed stdout (rc={self.proc.poll()})\n'
                                 f'--- stderr ---\n{self.stderr}')
        # Anything on stdout that is not one JSON object per line is a
        # protocol violation — json.loads raising here fails the test.
        frame = json.loads(line)
        assert isinstance(frame, dict), line
        return frame

    def handshake(self) -> dict:
        hello = self.read(timeout=60)
        assert hello['type'] == 'hello', hello
        self.send({'type': 'ready', 'host': 'subtitld', 'host_version': 'test', 'protocol': 1})
        return hello

    def read_until(self, req_id, *types, timeout=300.0):
        deadline = time.monotonic() + timeout
        seen = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f'never saw {types} for {req_id}; got '
                                     f'{[s.get("type") for s in seen]}\n--- stderr ---\n{self.stderr}')
            frame = self.read(timeout=remaining)
            if frame.get('id') != req_id:
                continue
            seen.append(frame)
            if frame.get('type') in types:
                return frame, seen

    def transcribe(self, req_id, params, timeout=600.0):
        self.send({'type': 'asr.transcribe', 'id': req_id, 'params': params})
        return self.read_until(req_id, 'result', 'error', timeout=timeout)

    def close(self, expect_rc=0) -> int:
        try:
            if self.proc.poll() is None:
                self.send({'type': 'shutdown'})
            rc = self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
            rc = self.proc.wait()
        finally:
            for stream in (self.proc.stdin,):
                try:
                    stream.close()
                except Exception:
                    pass
        if expect_rc is not None:
            assert rc == expect_rc, f'exit code {rc}\n--- stderr ---\n{self.stderr}'
            assert 'Traceback' not in self.stderr, self.stderr
        return rc

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def run_addon():
    started: list[_Addon] = []

    def _start(env_extra=None) -> _Addon:
        a = _Addon(env_extra)
        started.append(a)
        return a

    yield _start
    for a in started:
        a.kill()


def _silence(path: Path, seconds: float) -> Path:
    subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-y', '-f', 'lavfi',
                    '-i', 'anullsrc=r=44100:cl=stereo', '-t', str(seconds), str(path)],
                   check=True)
    return path


# ---------------------------------------------------------------------------
# Offline: configuration resolution
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_env(monkeypatch):
    for name in _SCRUB:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_option_beats_env_beats_default(clean_env):
    assert addon._resolve_model({}) == 'base'
    clean_env.setenv('WHISPERCPP_MODEL', 'small')
    assert addon._resolve_model({}) == 'small'
    assert addon._resolve_model({'model': 'tiny'}) == 'tiny'
    # Blank and None count as unset, exactly as the host treats them.
    assert addon._resolve_model({'model': ''}) == 'small'
    assert addon._resolve_model({'model': None}) == 'small'


def test_unknown_model_falls_back_to_base(clean_env):
    assert addon._resolve_model({'model': 'large-v9'}) == 'base'


def test_explicit_false_is_not_the_default(clean_env):
    key, env = 'phrase_grouping', 'WHISPERCPP_PHRASE_GROUPING'
    assert addon._opt_bool({}, key, env, True) is True
    assert addon._opt_bool({key: False}, key, env, True) is False
    assert addon._opt_bool({key: 'false'}, key, env, True) is False
    clean_env.setenv(env, '0')  # what the host exports for a bool False
    assert addon._opt_bool({}, key, env, True) is False
    assert addon._opt_bool({key: True}, key, env, True) is True
    clean_env.setenv(env, '1')
    assert addon._opt_bool({}, key, env, False) is True


@pytest.mark.parametrize('raw,expected', [
    ('0', 0), ('4', 4), (8, 8), ('', 0), (None, 0), ('many', 0), ('-3', 0), (' 2 ', 2),
])
def test_threads_resolution(clean_env, raw, expected):
    assert addon._resolve_threads({'threads': raw}) == expected


def test_threads_from_env(clean_env):
    clean_env.setenv('WHISPERCPP_THREADS', '2')
    assert addon._resolve_threads({}) == 2


@pytest.mark.parametrize('tag,expected', [
    ('en', 'en'), ('pt-br', 'pt'), ('pt-BR', 'pt'), ('en_US', 'en'), ('zh-Hans-CN', 'zh'),
    ('haw', 'haw'), ('jv', 'jw'), ('fil', 'tl'), ('nb-no', 'no'),
    ('cmn-hans-cn', 'zh'), ('cmn-Hant-TW', 'zh'), ('yue-hant-hk', 'zh'),
    ('', ''), (None, ''), ('auto', ''), ('und', ''),
])
def test_language_resolution(tag, expected):
    assert addon._resolve_language(tag) == expected


# Every value of Subtitld's `session.LANGUAGE_DICT_LIST` (the Import panel's
# language box), copied as data so the add-on tests never import the host.
HOST_LANGUAGE_VALUES = (
    'af-za', 'am-et', 'ar-ae', 'ar-bh', 'ar-dz', 'ar-eg', 'ar-il', 'ar-iq', 'ar-jo',
    'ar-kw', 'ar-lb', 'ar-ma', 'ar-om', 'ar-ps', 'ar-qa', 'ar-sa', 'ar-tn', 'az-az',
    'bg-bg', 'bn-bd', 'bn-in', 'ca-es', 'cmn-hans-cn', 'cmn-hans-hk', 'cmn-hant-tw',
    'cs-cz', 'da-dk', 'de-de', 'el-gr', 'en-au', 'en-ca', 'en-gb', 'en-gh', 'en-ie',
    'en-in', 'en-ke', 'en-ng', 'en-nz', 'en-ph', 'en-sg', 'en-tz', 'en-us', 'en-za',
    'es-ar', 'es-bo', 'es-cl', 'es-co', 'es-cr', 'es-do', 'es-ec', 'es-es', 'es-gt',
    'es-hn', 'es-mx', 'es-ni', 'es-pa', 'es-pe', 'es-pr', 'es-py', 'es-sv', 'es-us',
    'es-uy', 'es-ve', 'eu-es', 'fa-ir', 'fi-fi', 'fil-ph', 'fr-ca', 'fr-fr', 'gl-es',
    'gu-in', 'he-il', 'hi-in', 'hr-hr', 'hu-hu', 'hy-am', 'id-id', 'is-is', 'it-it',
    'ja-jp', 'jv-id', 'ka-ge', 'km-kh', 'kn-in', 'ko-kr', 'lo-la', 'lt-lt', 'lv-lv',
    'ml-in', 'mr-in', 'ms-my', 'nb-no', 'ne-np', 'nl-nl', 'pl-pl', 'pt-br', 'pt-pt',
    'ro-ro', 'ru-ru', 'si-lk', 'sk-sk', 'sl-si', 'sr-rs', 'su-id', 'sv-se', 'sw-ke',
    'sw-tz', 'ta-in', 'ta-lk', 'ta-my', 'ta-sg', 'te-in', 'th-th', 'tr-tr', 'uk-ua',
    'ur-in', 'ur-pk', 'vi-vn', 'yue-hant-hk', 'zu-za',
)
# Languages the host offers that Whisper has no model for.
HOST_LANGUAGES_WHISPER_LACKS = {'zu-za'}  # Zulu


def test_every_language_the_host_offers_resolves():
    """The host sends its tag verbatim and never filters engines by language,
    so a tag the add-on rejects makes that project language untranscribable
    (Chinese was, until `cmn`/`yue` were aliased)."""
    assert len(HOST_LANGUAGE_VALUES) == len(set(HOST_LANGUAGE_VALUES)) == 120
    rejected = set()
    for tag in HOST_LANGUAGE_VALUES:
        try:
            code = addon._resolve_language(tag)
        except addon._AddonError as exc:
            assert exc.code == addon.ERR_UNSUPPORTED_LANGUAGE
            rejected.add(tag)
        else:
            assert code in addon._LANGUAGE_SET, (tag, code)
    assert rejected == HOST_LANGUAGES_WHISPER_LACKS, (
        f'rejected host languages: {sorted(rejected)}; '
        f'only {sorted(HOST_LANGUAGES_WHISPER_LACKS)} lack a Whisper model')


@pytest.mark.parametrize('tag', ['xx', 'klingon', 'por', 'e'])
def test_unknown_language_is_an_error_not_a_guess(tag):
    with pytest.raises(addon._AddonError) as info:
        addon._resolve_language(tag)
    assert info.value.code == addon.ERR_UNSUPPORTED_LANGUAGE


def test_models_dir_layers(clean_env, tmp_path):
    import platformdirs
    default = Path(platformdirs.user_cache_dir('subtitld')) / 'models' / 'whispercpp'
    assert addon._models_dir() == default

    clean_env.setenv('SUBTITLD_MODELS_DIR', str(tmp_path / 'host'))
    assert addon._models_dir() == tmp_path / 'host' / 'whispercpp'

    clean_env.setenv('WHISPERCPP_MODELS_DIR', str(tmp_path / 'mine'))
    assert addon._models_dir() == tmp_path / 'mine'

    assert addon._models_dir({'models_dir': str(tmp_path / 'req')}) == tmp_path / 'req'
    assert addon._models_dir({'models_dir': '  '}) == tmp_path / 'mine'


def test_models_dir_accepts_a_file_inside_the_folder(clean_env, tmp_path):
    """The Configure dialog's `file` field can only browse to a file."""
    picked = tmp_path / 'ggml-base.bin'
    picked.write_bytes(b'lmgg')
    clean_env.setenv('WHISPERCPP_MODELS_DIR', str(picked))
    assert addon._models_dir() == tmp_path


def test_models_dir_default_matches_the_host(clean_env):
    """Layer 3 must be byte-for-byte the host's session.PATH_SUBTITLD_DATA_MODELS
    + 'whispercpp', i.e. where the built-in provider stored models."""
    import platformdirs
    host_models = Path(platformdirs.user_cache_dir('subtitld')) / 'models'  # session.py
    assert addon._models_dir() == host_models / 'whispercpp'
    if sys.platform.startswith('linux') and not os.environ.get('XDG_CACHE_HOME'):
        assert addon._models_dir() == Path.home() / '.cache' / 'subtitld' / 'models' / 'whispercpp'


def test_temp_root_falls_back_when_the_host_dir_is_gone(clean_env, tmp_path):
    clean_env.setenv('SUBTITLD_TEMP_DIR', str(tmp_path / 'gone'))
    assert addon._temp_root() == tempfile.gettempdir()
    clean_env.setenv('SUBTITLD_TEMP_DIR', str(tmp_path))
    assert addon._temp_root() == str(tmp_path)


def test_ffmpeg_lookup_order(clean_env, tmp_path):
    fake = tmp_path / 'my-ffmpeg'
    fake.write_bytes(b'')
    clean_env.setenv('FFMPEG_EXECUTABLE', str(fake))
    assert addon._find_ffmpeg() == str(fake)
    clean_env.setenv('SUBTITLD_FFMPEG_EXECUTABLE', str(fake) + '-host')
    assert addon._find_ffmpeg() == str(fake), 'a missing host path must not win'
    (tmp_path / 'my-ffmpeg-host').write_bytes(b'')
    assert addon._find_ffmpeg() == str(fake) + '-host'


def test_stale_partial_downloads_are_swept(tmp_path):
    target = tmp_path / 'ggml-base.bin'
    old = tmp_path / 'ggml-base.bin.123.part'
    legacy = tmp_path / 'ggml-base.bin.part'  # the built-in's name
    fresh = tmp_path / 'ggml-base.bin.456.part'
    other = tmp_path / 'ggml-small.bin.1.part'
    for p in (old, legacy, fresh, other):
        p.write_bytes(b'x')
    ancient = time.time() - addon._STALE_PART_SEC - 60
    for p in (old, legacy, other):
        os.utime(p, (ancient, ancient))
    addon._sweep_stale_parts(target)
    assert not old.exists() and not legacy.exists()
    assert fresh.exists(), 'a live download must not be touched'
    assert other.exists(), 'only this model\'s partials are swept'


# ---------------------------------------------------------------------------
# Offline: segments and regrouping
# ---------------------------------------------------------------------------
class _Seg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_segment_centiseconds():
    assert addon._segment_to_dict(_Seg(t0=150, t1=420, text=' Hi. ')) == {
        'start': 1.5, 'end': 4.2, 'text': 'Hi.', 'speaker': 'A'}


def test_segment_seconds_fallback_and_empty_text():
    assert addon._segment_to_dict(_Seg(start=1.0, end=2.0, text='x'))['end'] == 2.0
    assert addon._segment_to_dict(_Seg(t0=0, t1=10, text='   ')) is None


def test_regroup_splits_sentences_and_merges_fragments():
    raw = [
        {'start': 0.0, 'end': 2.0, 'text': 'Hello there. How are', 'speaker': 'A'},
        {'start': 2.0, 'end': 3.0, 'text': 'you today?', 'speaker': 'A'},
    ]
    out = addon._regroup_into_phrases(raw)
    assert [p['text'] for p in out] == ['Hello there.', 'How are you today?']
    assert out[0]['start'] == 0.0
    assert out[1]['end'] == 3.0
    assert raw[0]['text'] == 'Hello there. How are', 'input must not be mutated'


def test_regroup_respects_the_readability_cap():
    words = [{'start': float(i), 'end': float(i) + 1, 'text': 'word ' * 6, 'speaker': 'A'}
             for i in range(10)]
    out = addon._regroup_into_phrases(words)
    assert len(out) > 1
    assert all(len(p['text']) <= addon._PHRASE_MAX_CHARS for p in out)
    assert all(p['end'] - p['start'] <= addon._PHRASE_MAX_DURATION_SEC for p in out)


def test_regroup_never_merges_speakers_and_keeps_quotes():
    out = addon._regroup_into_phrases([
        {'start': 0, 'end': 1, 'text': 'She said "hi."', 'speaker': 'A'},
        {'start': 1, 'end': 2, 'text': 'and then', 'speaker': 'A'},
        {'start': 2, 'end': 3, 'text': 'left', 'speaker': 'B'},
    ])
    assert [p['text'] for p in out] == ['She said "hi."', 'and then', 'left']
    assert addon._regroup_into_phrases([]) == []


# ---------------------------------------------------------------------------
# Offline: error mapping
# ---------------------------------------------------------------------------
def test_error_mapping():
    def http(code):
        return urllib.error.HTTPError('u', code, 'x', {}, None)

    assert addon._classify(http(404))[0] == addon.ERR_MODEL_MISSING
    code, _, retry = addon._classify(http(503))
    assert (code, retry) == (addon.ERR_NETWORK_UNAVAILABLE, True)
    code, msg, retry = addon._classify(urllib.error.URLError('refused'))
    assert (code, retry) == (addon.ERR_NETWORK_UNAVAILABLE, True) and 'offline' in msg
    assert addon._classify(OSError(errno.ENOSPC, 'full'))[0] == addon.ERR_DISK_FULL
    assert addon._classify(MemoryError())[0] == addon.ERR_OOM
    assert addon._classify(ValueError('boom'))[0] == addon.ERR_INTERNAL


def test_native_log_callback_never_raises_and_captures():
    addon._on_native_log('not-a-level', None)
    addon._on_native_log(4, 'whisper_init: failed to load model\n', object())
    sink = []
    addon._native_capture = sink
    try:
        addon._on_native_log(4, 'failed to load model\n')
        addon._on_native_log(2, 'loading model\n')
        addon._on_native_log(5, ' (continued)\n')  # CONT inherits INFO
        addon._on_native_log(2, '\n')
    finally:
        addon._native_capture = None
    assert sink == [(4, 'failed to load model'), (2, 'loading model'), (2, ' (continued)')]


_GOOD_LOAD = [
    (2, "whisper_init_from_file_with_params_no_state: loading model from 'x'"),
    (2, 'whisper_model_load: model size    =   42.82 MB'),
    (2, 'whisper_init_state: kv self size  =    3.15 MB'),
    (2, 'whisper_init_state: compute buffer (decode) =   95.91 MB'),
]


def test_load_verdict_accepts_a_completed_init_even_with_backend_errors():
    """A GPU backend that fails to start logs an ERROR and whisper.cpp falls
    back to the CPU. That model is fine and must not be rejected."""
    lines = [(4, 'whisper_backend_init_gpu: failed to initialize Metal backend')] + _GOOD_LOAD
    assert addon._load_failure(lines, Path('/m/ggml-base.bin')) is None


def test_load_verdict_rejects_an_incomplete_init():
    lines = [(2, 'whisper_model_load: type = 1 (tiny)'),
             (4, 'whisper_model_load: ERROR not all tensors loaded from model file - expected 167, got 3'),
             (4, 'whisper_init_with_params_no_state: failed to load model')]
    err = addon._load_failure(lines, Path('/m/ggml-tiny.bin'))
    assert err.code == addon.ERR_MODEL_MISSING
    assert 'failed to load model' in err.message and '/m/ggml-tiny.bin' in err.message
    assert addon._load_failure([], Path('/m/x.bin')).code == addon.ERR_MODEL_MISSING


def test_load_verdict_reports_allocation_failure_as_oom():
    lines = _GOOD_LOAD[:3] + [(4, 'whisper_init_state: failed to allocate memory for the decoder')]
    assert addon._load_failure(lines, Path('/m/ggml-large-v3.bin')).code == addon.ERR_OOM


# ---------------------------------------------------------------------------
# Protocol, over a real subprocess
# ---------------------------------------------------------------------------
def test_hello_advertises_asr_and_all_languages(run_addon):
    a = run_addon()
    hello = a.handshake()
    assert hello['addon'] == 'whispercpp'
    assert hello['protocol'] == 1
    assert hello['version'] == MANIFEST['version']
    (cap,) = hello['capabilities']
    assert cap['task'] == 'asr.transcribe'
    assert cap['languages'] == MANIFEST['languages']
    assert cap['streaming'] is False
    a.close()


def test_eof_on_stdin_exits_cleanly(run_addon):
    a = run_addon()
    a.handshake()
    a.proc.stdin.close()
    assert a.proc.wait(timeout=20) == 0
    assert 'Traceback' not in a.stderr


@pytest.mark.parametrize('params,needle', [
    ({}, 'audio_path is required'),
    ({'audio_path': ''}, 'audio_path is required'),
    ({'audio_path': '/definitely/not/here.wav'}, 'not a file'),
])
def test_bad_audio_path_is_rejected(run_addon, params, needle):
    a = run_addon()
    a.handshake()
    err, _ = a.transcribe('r1', params, timeout=30)
    assert err['type'] == 'error'
    assert err['code'] == addon.ERR_BAD_PARAMS
    assert needle in err['message']
    a.close()


def test_unsupported_language_is_rejected_before_any_work(run_addon, tmp_path):
    a = run_addon({'WHISPERCPP_MODELS_DIR': str(tmp_path / 'models')})
    a.handshake()
    err, seen = a.transcribe('r1', {'audio_path': str(JFK), 'language': 'tlh'}, timeout=30)
    assert err['code'] == addon.ERR_UNSUPPORTED_LANGUAGE
    assert [f['type'] for f in seen] == ['error'], 'no progress before rejecting'
    assert not (tmp_path / 'models').exists()
    a.close()


def test_missing_ffmpeg_is_a_clear_error(run_addon, tmp_path):
    empty = tmp_path / 'empty-path'
    empty.mkdir()
    a = run_addon({'PATH': str(empty),
                   'SUBTITLD_FFMPEG_EXECUTABLE': str(tmp_path / 'nope'),
                   'FFMPEG_EXECUTABLE': ''})
    a.handshake()
    err, _ = a.transcribe('r1', {'audio_path': str(JFK), 'language': 'en'}, timeout=30)
    assert err['code'] == addon.ERR_BAD_PARAMS
    assert 'ffmpeg' in err['message']
    a.close()


def test_unknown_frame_still_answers(run_addon):
    a = run_addon()
    a.handshake()
    a.send({'type': 'tts.synthesize', 'id': 'r9', 'params': {}})
    err, _ = a.read_until('r9', 'error', timeout=30)
    assert err['code'] == addon.ERR_BAD_PARAMS
    a.send({'type': 'asr.transcribe', 'id': 'r10', 'params': ['not', 'a', 'dict']})
    err, _ = a.read_until('r10', 'error', timeout=30)
    assert err['code'] == addon.ERR_BAD_PARAMS
    a.send({'type': 'cancel', 'target': 'no-such-request'})
    a.send({'type': 'garbage'})
    a.proc.stdin.write(b'this is not json\n\n')
    a.proc.stdin.flush()
    a.close()


# ---------------------------------------------------------------------------
# ffmpeg + download path, against a local HTTP server and a fake model
# ---------------------------------------------------------------------------
class _ModelServer:
    """Serves /ggml-<model>.bin from a dict, optionally slowly."""

    def __init__(self, files: dict[str, bytes], delay: float = 0.0, chunk: int = 65536):
        outer = self
        self.files = files
        self.hits: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits.append(self.path)
                assert self.headers.get('User-Agent') == 'subtitld-whispercpp/1'
                body = outer.files.get(self.path.lstrip('/'))
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    for i in range(0, len(body), chunk):
                        self.wfile.write(body[i:i + chunk])
                        self.wfile.flush()
                        if delay:
                            time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


FAKE_MODEL = b'lmgg' + b'\0' * (3 * 1024 * 1024)


@needs_ffmpeg
def test_download_then_corrupt_model_fails_cleanly_without_crashing(run_addon, tmp_path):
    """A model whisper.cpp cannot load must become an error frame. pywhispercpp
    hides the NULL context, so without the log check the next decode segfaults
    and the host only ever sees `process_exited`."""
    server = _ModelServer({'ggml-tiny.bin': FAKE_MODEL})
    models = tmp_path / 'models'
    temp = tmp_path / 'temp'
    temp.mkdir()
    try:
        a = run_addon({'WHISPERCPP_ADDON_MODEL_BASE_URL': server.url,
                       'WHISPERCPP_MODELS_DIR': str(models),
                       'SUBTITLD_TEMP_DIR': str(temp)})
        a.handshake()
        err, seen = a.transcribe('r1', {'audio_path': str(JFK), 'language': 'en',
                                        'options': {'model': 'tiny'}}, timeout=120)
        assert err['type'] == 'error', err
        assert err['code'] == addon.ERR_MODEL_MISSING
        assert 'ggml-tiny.bin' in err['message'] and 'delete it' in err['message']

        progress = [f for f in seen if f['type'] == 'progress']
        values = [f['value'] for f in progress]
        assert values == sorted(values), 'progress must never go backwards'
        assert any('Downloading tiny model' in f['message'] for f in progress)
        assert max(values) >= 0.9 - 1e-9, 'download spans 0.05 -> 0.90'

        assert (models / 'ggml-tiny.bin').read_bytes() == FAKE_MODEL
        assert list(models.glob('*.part')) == [], 'partial file left behind'
        assert list(temp.iterdir()) == [], 'temporary WAV left behind'
        assert server.hits == ['/ggml-tiny.bin']

        # Still alive, and the cached file is reused (no second download).
        err2, _ = a.transcribe('r2', {'audio_path': str(JFK), 'language': 'en',
                                      'options': {'model': 'tiny'}}, timeout=120)
        assert err2['code'] == addon.ERR_MODEL_MISSING
        assert server.hits == ['/ggml-tiny.bin']
        a.close()
    finally:
        server.close()


@needs_ffmpeg
def test_non_model_download_is_rejected_and_discarded(run_addon, tmp_path):
    server = _ModelServer({'ggml-base.bin': b'<!DOCTYPE html>' + b' ' * 5000})
    models = tmp_path / 'models'
    try:
        a = run_addon({'WHISPERCPP_ADDON_MODEL_BASE_URL': server.url,
                       'WHISPERCPP_MODELS_DIR': str(models)})
        a.handshake()
        err, _ = a.transcribe('r1', {'audio_path': str(JFK)}, timeout=120)
        assert err['code'] == addon.ERR_MODEL_MISSING
        assert 'did not return a whisper.cpp model' in err['message']
        assert list(models.iterdir()) == []
        a.close()
    finally:
        server.close()


@needs_ffmpeg
def test_existing_non_model_file_is_reported_not_loaded(run_addon, tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'ggml-base.bin').write_bytes(b'garbage' * 10)
    a = run_addon({'WHISPERCPP_MODELS_DIR': str(models),
                   'WHISPERCPP_ADDON_MODEL_BASE_URL': 'http://127.0.0.1:9'})
    a.handshake()
    err, _ = a.transcribe('r1', {'audio_path': str(JFK)}, timeout=60)
    assert err['code'] == addon.ERR_MODEL_MISSING
    assert 'not a whisper.cpp model file' in err['message']
    assert (models / 'ggml-base.bin').exists(), 'never delete a user file'
    a.close()


@needs_ffmpeg
def test_cancel_during_download(run_addon, tmp_path):
    server = _ModelServer({'ggml-small.bin': b'lmgg' + b'\0' * (40 * 1024 * 1024)},
                          delay=0.05)
    models = tmp_path / 'models'
    try:
        a = run_addon({'WHISPERCPP_ADDON_MODEL_BASE_URL': server.url,
                       'WHISPERCPP_MODELS_DIR': str(models)})
        a.handshake()
        a.send({'type': 'asr.transcribe', 'id': 'r1',
                'params': {'audio_path': str(JFK), 'options': {'model': 'small'}}})
        while True:
            frame, _ = a.read_until('r1', 'progress', 'result', 'error', timeout=60)
            assert frame['type'] == 'progress', frame
            if 'MB' in frame['message']:
                break
        t0 = time.monotonic()
        a.send({'type': 'cancel', 'target': 'r1'})
        err, seen = a.read_until('r1', 'result', 'error', timeout=30)
        assert err['type'] == 'error' and err['code'] == addon.ERR_CANCELLED
        assert time.monotonic() - t0 < 5
        assert not (models / 'ggml-small.bin').exists()
        assert list(models.glob('*.part')) == [], 'cancelled download left its partial'
        a.close()
    finally:
        server.close()


@needs_ffmpeg
def test_unreachable_model_host_is_retryable(run_addon, tmp_path):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]  # nothing listens here once closed
    a = run_addon({'WHISPERCPP_ADDON_MODEL_BASE_URL': f'http://127.0.0.1:{port}',
                   'WHISPERCPP_MODELS_DIR': str(tmp_path / 'models')})
    a.handshake()
    err, _ = a.transcribe('r1', {'audio_path': str(JFK)}, timeout=60)
    assert err['code'] == addon.ERR_NETWORK_UNAVAILABLE
    assert err['retryable'] is True
    a.close()


@needs_ffmpeg
def test_undecodable_input_is_reported(run_addon, tmp_path):
    junk = tmp_path / 'junk.mp3'
    junk.write_bytes(b'not audio at all' * 100)
    temp = tmp_path / 'temp'
    temp.mkdir()
    a = run_addon({'SUBTITLD_TEMP_DIR': str(temp)})
    a.handshake()
    err, _ = a.transcribe('r1', {'audio_path': str(junk)}, timeout=60)
    assert err['code'] == addon.ERR_INTERNAL
    assert 'ffmpeg' in err['message']
    assert list(temp.iterdir()) == []
    a.close()


@needs_ffmpeg
def test_too_short_audio_needs_no_model(run_addon, tmp_path):
    clip = _silence(tmp_path / 'blip.wav', 0.05)
    a = run_addon({'WHISPERCPP_MODELS_DIR': str(tmp_path / 'models'),
                   'WHISPERCPP_ADDON_MODEL_BASE_URL': 'http://127.0.0.1:9'})
    a.handshake()
    res, _ = a.transcribe('r1', {'audio_path': str(clip)}, timeout=60)
    assert res['type'] == 'result', res
    assert res['data']['segments'] == []
    a.close()


# ---------------------------------------------------------------------------
# Real transcription — opt-in, downloads ggml-tiny.bin once
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def tiny_models_dir():
    root = Path(os.environ.get('WHISPERCPP_TEST_MODELS_DIR')
                or Path(tempfile.gettempdir()) / 'whispercpp-addon-test-models')
    model = root / 'ggml-tiny.bin'
    if not model.is_file():
        # Let the add-on fetch it: that is the real Hugging Face download path.
        a = _Addon({'WHISPERCPP_MODELS_DIR': str(root)})
        try:
            a.handshake()
            frame, seen = a.transcribe('warm', {'audio_path': str(JFK), 'language': 'en',
                                                'options': {'model': 'tiny'}}, timeout=900)
            assert frame['type'] == 'result', frame
            assert any('Downloading tiny model' in f.get('message', '') for f in seen)
            a.close()
        finally:
            a.kill()
    assert model.is_file() and model.read_bytes()[:4] == b'lmgg'
    assert list(root.glob('*.part')) == []
    return root


def _tiny_env(models_dir, **extra):
    env = {'WHISPERCPP_MODELS_DIR': str(models_dir),
           # A download attempt here would mean the cache was not used.
           'WHISPERCPP_ADDON_MODEL_BASE_URL': 'http://127.0.0.1:9',
           'WHISPERCPP_MODEL': 'tiny'}
    env.update(extra)
    return env


@transcription
def test_real_transcription(run_addon, tiny_models_dir, tmp_path):
    temp = tmp_path / 'temp'
    temp.mkdir()
    a = run_addon(_tiny_env(tiny_models_dir, SUBTITLD_TEMP_DIR=str(temp)))
    a.handshake()
    res, seen = a.transcribe('r1', {'audio_path': str(JFK), 'language': 'en-us'})
    assert res['type'] == 'result', res
    data = res['data']
    text = ' '.join(s['text'] for s in data['segments'])
    assert 'country' in text.lower(), text
    assert 'americans' in text.lower(), text
    assert data['model'] == 'tiny'
    assert data['duration_sec'] == pytest.approx(11.0, abs=0.05)
    for seg in data['segments']:
        assert set(seg) == {'start', 'end', 'text', 'speaker'}
        assert 0.0 <= seg['start'] <= seg['end'] <= 11.5

    partials = [f for f in seen if f['type'] == 'partial']
    assert partials, 'segments must stream as partial frames'
    assert all('segment' in f['data'] for f in partials)
    progress = [f['value'] for f in seen if f['type'] == 'progress']
    assert progress == sorted(progress), progress
    assert progress[-1] == 1.0
    assert list(temp.iterdir()) == [], 'temporary WAV left behind'
    a.close()


@transcription
def test_auto_detect_and_raw_segmentation(run_addon, tiny_models_dir):
    a = run_addon(_tiny_env(tiny_models_dir, WHISPERCPP_PHRASE_GROUPING='0'))
    a.handshake()
    res, seen = a.transcribe('r1', {'audio_path': str(JFK), 'language': ''})
    assert res['type'] == 'result', res
    assert 'country' in ' '.join(s['text'] for s in res['data']['segments']).lower()
    # With grouping off the result is exactly what streamed.
    partials = [f['data']['segment'] for f in seen if f['type'] == 'partial']
    assert res['data']['segments'] == partials
    a.close()


@transcription
def test_model_stays_loaded_and_short_input_never_returns_stale_text(run_addon, tiny_models_dir, tmp_path):
    """Record mode sends one request per phrase: the model must load once.
    And whisper.cpp keeps the previous results on its early-return path, so a
    tiny clip after a real one must come back empty, not with JFK's words."""
    blip = _silence(tmp_path / 'blip.wav', 0.05)
    quiet = _silence(tmp_path / 'quiet.wav', 1.5)
    a = run_addon(_tiny_env(tiny_models_dir, WHISPERCPP_THREADS='2'))
    a.handshake()
    first, seen1 = a.transcribe('r1', {'audio_path': str(JFK), 'language': 'en'})
    assert first['type'] == 'result', first
    second, seen2 = a.transcribe('r2', {'audio_path': str(JFK), 'language': 'en',
                                        'options': {'threads': '0'}})
    assert second['type'] == 'result', second
    third, _ = a.transcribe('r3', {'audio_path': str(blip), 'language': 'en'})
    assert third['type'] == 'result' and third['data']['segments'] == []
    fourth, _ = a.transcribe('r4', {'audio_path': str(quiet), 'language': 'en'})
    assert fourth['type'] == 'result', fourth
    assert 'country' not in json.dumps(fourth['data']).lower()

    def loads(seen):
        return [f for f in seen if f['type'] == 'progress' and 'Loading' in f['message']]

    assert len(loads(seen1)) == 1
    assert loads(seen2) == [], 'the model was reloaded for the second request'
    assert 'country' in json.dumps(second['data']).lower()
    a.close()
    assert a.stderr.count('loaded ggml-tiny.bin') == 1, a.stderr


@transcription
def test_cancel_mid_transcription(run_addon, tiny_models_dir, tmp_path):
    long = tmp_path / 'long.wav'
    subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-y', '-stream_loop', '30',
                    '-i', str(JFK), '-ac', '1', '-ar', '16000', str(long)], check=True)
    a = run_addon(_tiny_env(tiny_models_dir))
    a.handshake()
    a.send({'type': 'asr.transcribe', 'id': 'r1',
            'params': {'audio_path': str(long), 'language': 'en'}})
    frame, _ = a.read_until('r1', 'partial', 'result', 'error', timeout=300)
    assert frame['type'] == 'partial', frame
    t0 = time.monotonic()
    a.send({'type': 'cancel', 'target': 'r1'})
    err, seen = a.read_until('r1', 'result', 'error', timeout=60)
    elapsed = time.monotonic() - t0
    assert err['type'] == 'error', 'a cancelled decode must not yield a result'
    assert err['code'] == addon.ERR_CANCELLED
    assert elapsed < 10, f'cancel took {elapsed:.1f}s: abort_callback not honoured'

    # The context is still usable afterwards.
    res, _ = a.transcribe('r2', {'audio_path': str(JFK), 'language': 'en'})
    assert res['type'] == 'result' and 'country' in json.dumps(res['data']).lower()
    a.close()


@transcription
def test_shutdown_during_transcription_is_prompt(run_addon, tiny_models_dir, tmp_path):
    long = tmp_path / 'long.wav'
    subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-y', '-stream_loop', '30',
                    '-i', str(JFK), '-ac', '1', '-ar', '16000', str(long)], check=True)
    a = run_addon(_tiny_env(tiny_models_dir))
    a.handshake()
    a.send({'type': 'asr.transcribe', 'id': 'r1',
            'params': {'audio_path': str(long), 'language': 'en'}})
    frame, _ = a.read_until('r1', 'partial', 'result', 'error', timeout=300)
    assert frame['type'] == 'partial', frame
    t0 = time.monotonic()
    assert a.close() == 0
    assert time.monotonic() - t0 < 5, 'the host SIGTERMs after 5 s'
