# Whisper (offline) add-on for Subtitld

Offline **speech-to-text** for [Subtitld](https://github.com/Subtitld/subtitld),
powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through
[pywhispercpp](https://github.com/absadiki/pywhispercpp).

- Serves the `asr.transcribe` task over Subtitld's add-on protocol.
- **CPU only, no torch, no cloud account.** 99 languages, or auto-detect.
- Segments appear on the timeline while decoding, then are regrouped into one
  subtitle row per sentence.
- This is Subtitld's former built-in `whispercpp` engine, moved into an
  add-on. It keeps the same id, settings and model files, so an existing
  setup carries over unchanged.

## Models

Models are not in the archive. The one you pick downloads once, on first use,
from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)
on Hugging Face. After that, transcription is fully offline.

| Model | Download | RAM (approx.) | Notes |
|-------|----------|---------------|-------|
| `tiny`, `tiny.en` | 75 MB | ~0.3 GB | fastest, roughest |
| `base`, `base.en` | 142 MB | ~0.4 GB | **default** |
| `small`, `small.en` | 466 MB | ~0.9 GB | |
| `medium`, `medium.en` | 1.5 GB | ~2.1 GB | |
| `large-v3` | 2.9 GB | ~3.9 GB | most accurate, slowest |

`.en` models are English-only and slightly faster. All of them are MIT
licensed (OpenAI Whisper weights converted to GGML).

### Where models are stored

Models are kept in Subtitld's shared model cache, **never** inside the add-on
folder, because Subtitld replaces that folder on every add-on update. The
first match wins:

1. The **Models folder** option (`WHISPERCPP_MODELS_DIR`).
2. `$SUBTITLD_MODELS_DIR/whispercpp`, when the host exports it.
3. `<user cache>/subtitld/models/whispercpp`, which is where Subtitld's
   built-in engine stored them:
   - Linux: `~/.cache/subtitld/models/whispercpp`
   - macOS: `~/Library/Caches/subtitld/models/whispercpp`
   - Windows: `%LOCALAPPDATA%\subtitld\subtitld\Cache\models\whispercpp`

Files are named `ggml-<model>.bin`, the same names the built-in engine used,
so models you already downloaded are reused with no download. Downloads go to
a `.part` file and are renamed only once complete.

Uninstalling the add-on does **not** delete downloaded models. Remove the
folder above yourself to reclaim the space.

## Options

Set these in **Settings ▸ Add-ons ▸ Whisper (offline) ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| Model | `base` | `WHISPERCPP_MODEL` | See the table above. |
| CPU threads | `0` | `WHISPERCPP_THREADS` | `0` = auto (whisper.cpp uses up to 4). |
| Group output by phrases | on | `WHISPERCPP_PHRASE_GROUPING` | Off keeps whisper's raw segmentation. |
| Models folder | *(empty)* | `WHISPERCPP_MODELS_DIR` | Empty = shared cache. You can type a folder, or browse to any file inside it. |

Each request resolves every option in this order: the request's own
`options`, then the env var (Subtitld sets these from the Configure dialog
when it starts the add-on), then the default.

Other environment variables:

| Variable | Purpose |
|----------|---------|
| `SUBTITLD_FFMPEG_EXECUTABLE`, `FFMPEG_EXECUTABLE` | ffmpeg to use, checked in that order before `PATH`. |
| `SUBTITLD_TEMP_DIR` | Where the temporary 16 kHz WAV goes (default: system temp). It is always deleted. |
| `SUBTITLD_MODELS_DIR` | Host model cache root (see above). |
| `WHISPERCPP_ADDON_LOG_LEVEL` | `DEBUG` also shows whisper.cpp's own log. Default `INFO`. |
| `WHISPERCPP_ADDON_MODEL_BASE_URL` | Model download mirror. Used by the tests. |
| `SSL_CERT_FILE`, `SSL_CERT_DIR` | Extra CA certificates for the model download (added to the bundled ones). |

## Requirements

- **ffmpeg**, used to decode audio and video. Subtitld provides its own copy.
  If you run the add-on outside Subtitld, put ffmpeg on `PATH`.
- Disk space for the model you pick, plus about 75 MB for the add-on.

## Behaviour notes

- **Record mode:** the model stays loaded between requests. Subtitld sends one
  request per spoken phrase, so only the first one pays the load time.
  Switching models unloads the previous one first.
- **Cancel** stops a running transcription within about a second, including
  one in the middle of decoding. A cancelled request never returns partial
  text as a finished result.
- **Language:** tags like `pt-br` or `en_US` use their primary subtag.
  Subtitld's Chinese tags (`cmn-hans-cn`, `cmn-hant-tw`, `yue-hant-hk`, ...)
  map to Whisper's `zh`. An empty language means auto-detect. A language
  Whisper doesn't support (of Subtitld's list, only Zulu) is rejected with
  `unsupported_language`.
- **Downloads** use HTTPS, trusting both the system's certificate store and
  the certifi bundle shipped inside the add-on, so they work on systems whose
  store the frozen binary cannot find (Flatpak, Fedora, Arch, macOS). If your
  network inspects HTTPS through a proxy, point `SSL_CERT_FILE` at a file
  with its certificate authority. A download that ends early is discarded,
  also when the server sends no file size.
- **Errors** use Subtitld's protocol codes: `model_missing` (including a
  corrupt model file, with the path to delete), `network_unavailable`
  (retryable), `disk_full`, `oom`, `unsupported_language`, `bad_params`,
  `cancelled` and `internal`.

## Platform support

| Platform | Archive | Tested |
|----------|---------|--------|
| Linux x86-64 | `whispercpp-<version>-linux-x86_64.zip` | Built and tested locally and in CI |
| macOS arm64 | `whispercpp-<version>-macos-arm64.zip` | CI only |
| Windows x86-64 | `whispercpp-<version>-windows-x86_64.zip` | CI only |

Every platform's frozen binary, unpacked from its zip the way Subtitld
installs it, runs the full subprocess test suite in CI before release. That
includes a real transcription with `ggml-tiny.bin`, downloaded through the
binary with the runner's own CA certificates hidden.
pywhispercpp 1.5.1 has no Intel macOS wheel, so there is no Intel Mac build.

Archive layout:

```
manifest.json                 executable: bin/whispercpp-addon (.exe on Windows)
LICENSE                       Apache-2.0
README.md
THIRD_PARTY_LICENSES/         licences of everything frozen into bin/
bin/whispercpp-addon          PyInstaller onedir launcher (+ bin/_internal/)
```

## Development

Use a dedicated virtual environment. pywhispercpp pulls in numpy, which you
don't want upgraded inside another project's environment.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q                                            # needs ffmpeg for most tests
WHISPERCPP_TEST_TRANSCRIPTION=1 pytest -q            # + real inference (downloads ggml-tiny.bin once)
```

The real-inference tests keep `ggml-tiny.bin` in `WHISPERCPP_TEST_MODELS_DIR`
(default `<tmp>/whispercpp-addon-test-models`), never in Subtitld's cache.
`tests/data/jfk.flac` is whisper.cpp's `samples/jfk.wav` (public domain).

Build and test the frozen binary:

```bash
pip install -e '.[build,dev]'
pyinstaller pyinstaller.spec --noconfirm
WHISPERCPP_TEST_TRANSCRIPTION=1 \
WHISPERCPP_TEST_BINARY="$PWD/dist/whispercpp-addon/whispercpp-addon" \
  pytest -q tests/test_addon.py                      # the path must be absolute
```

Package it the way CI does:

```bash
python tools/fetch_third_party_licenses.py --out release-staging/THIRD_PARTY_LICENSES
mkdir -p release-staging/bin && cp -R dist/whispercpp-addon/. release-staging/bin/
cp manifest.json LICENSE README.md release-staging/
python tools/make_archive.py release-staging whispercpp-0.0.1-linux-x86_64.zip
```

`tools/make_archive.py` stores symlinks as regular files, because Subtitld's
installer cannot recreate links. `pyinstaller.spec` is commented throughout.
Read it before changing it: most of what it guards against fails at the first
transcription, not at build time.

## Releasing

1. Bump the version in `manifest.json`, `pyproject.toml`,
   `whispercpp_addon/__init__.py` and `ADDON_VERSION` in
   `whispercpp_addon/__main__.py`. A test fails if they disagree.
2. Push a `v<version>` tag. `.github/workflows/release.yml` runs the tests,
   builds and tests all three platforms, checks each archive's contents, and
   publishes the zips with `.sha256` sidecars and a source tarball. The job
   refuses to publish if the tag doesn't match `manifest.json` or a platform
   is missing.

## Licence

The add-on code is Apache-2.0 (see [LICENSE](LICENSE)). whisper.cpp and
pywhispercpp are MIT. Release archives include `THIRD_PARTY_LICENSES/`,
which covers everything frozen into the binary: numpy with OpenBLAS, the
Python runtime, the GCC runtime on Linux and the MSVC runtime on Windows.
