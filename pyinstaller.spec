# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld whisper.cpp add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/whispercpp-addon/` (launcher + `_internal/`). The release
workflow copies it to `bin/` inside the archive, next to manifest.json.

Every directive is annotated with the failure it prevents. Most of them fail
at FIRST TRANSCRIPTION, not at build time: the build succeeds, the handshake
succeeds (it never touches the native libraries' dependencies), and only a
real decode reveals a missing library — which is why CI runs one through the
frozen binary on every platform.

Run from the add-on root:

    pyinstaller pyinstaller.spec --noconfirm
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()


# ---------------------------------------------------------------------------
# Fail the build, loudly, if the runtime is not installed. Otherwise
# PyInstaller happily freezes a binary whose import guard reports
# `hello_error` on every machine.
# ---------------------------------------------------------------------------
# pywhispercpp imports numpy at module scope (the binding takes numpy arrays)
# and requests/tqdm in pywhispercpp/utils.py, so none of them can be excluded
# even though the add-on never uses pywhispercpp's own downloader. certifi is
# the add-on's own: the model download trusts its CA bundle
# (`_internal/certifi/cacert.pem`, collected by PyInstaller's certifi hook),
# because the bundled OpenSSL cannot find the user's system store.
_REQUIRED = ('pywhispercpp', '_pywhispercpp', 'numpy', 'platformdirs',
             'requests', 'tqdm', 'certifi')
_missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]
if _missing:
    raise SystemExit(f'FATAL: required package(s) not installed: {", ".join(_missing)}. '
                     f'Run `pip install -e .[build]` first.')


# ---------------------------------------------------------------------------
# The whisper.cpp / ggml shared libraries.
#
# `collect_dynamic_libs('pywhispercpp')` returns NOTHING, on every platform:
# the extension `_pywhispercpp` is a top-level module installed at the
# site-packages ROOT, and its libraries sit beside it rather than inside the
# `pywhispercpp` package. On Linux they only land in the bundle as a side
# effect of PyInstaller's ELF analysis of the extension — an accident, not a
# guarantee. So they are collected explicitly, from the wheel's own RECORD:
#
#   Linux   pywhispercpp.libs/*.so*   auditwheel-mangled, found through the
#           extension's RPATH=$ORIGIN/pywhispercpp.libs. Includes libgomp.
#           The wheel ALSO installs unmangled copies (libwhisper.so,
#           libggml*.so*, ~6.6 MB) at the site-packages root that nothing
#           links against; those are left out.
#   macOS   pywhispercpp/.dylibs/*.dylib   delocate'd; the extension loads
#           them via @loader_path/pywhispercpp/.dylibs/. They MUST travel as
#           `binaries`: `.dylib` is not an import suffix, so
#           `collect_data_files` would sweep them into `datas`, where
#           PyInstaller copies them without rewriting load commands or
#           re-signing — and the first Model() dies on an arm64 machine.
#           The root-level duplicates are left out here too.
#   Windows every .dll at the site-packages root (delvewheel-mangled, no
#           .libs dir): whisper, ggml, ggml-base, ggml-cpu, AND msvcp140 +
#           vcomp140. A whisper*/ggml*-only glob omits the OpenMP runtime and
#           yields "DLL load failed while importing _pywhispercpp" on any
#           machine without the VC++ redistributable. pywhispercpp's
#           delvewheel patch adds the directory above the package (the
#           bundle's `_internal/`) to the DLL search path, which is exactly
#           where these land.
# ---------------------------------------------------------------------------
_UNMANGLED_ROOT_LIB = re.compile(r'^lib(whisper|ggml[\w-]*)(\.\d+)*\.(so(\.\d+)*|dylib)$')


def _whispercpp_binaries() -> list[tuple[str, str]]:
    from importlib.metadata import distribution

    dist = distribution('pywhispercpp')
    site = Path(dist.locate_file(''))
    found: list[tuple[str, str]] = []
    for rel in dist.files or []:
        parts = Path(str(rel)).parts
        name = parts[-1]
        src = Path(dist.locate_file(rel))
        if not src.is_file():
            continue
        if sys.platform.startswith('linux'):
            if parts[0] == 'pywhispercpp.libs' and '.so' in name:
                found.append((str(src), 'pywhispercpp.libs'))
        elif sys.platform == 'darwin':
            if parts[:2] == ('pywhispercpp', '.dylibs') and name.endswith('.dylib'):
                found.append((str(src), 'pywhispercpp/.dylibs'))
        elif sys.platform == 'win32':
            if len(parts) == 1 and name.lower().endswith('.dll'):
                found.append((str(src), '.'))

    names = [Path(src).name.lower() for src, _ in found]
    required = ['whisper', 'ggml-base', 'ggml-cpu']
    if sys.platform.startswith('linux'):
        required.append('libgomp')
    elif sys.platform == 'win32':
        required += ['vcomp140', 'msvcp140']
    missing = [r for r in required if not any(n.startswith(r) or n.startswith('lib' + r)
                                              for n in names)]
    if missing:
        raise SystemExit(f'FATAL: pywhispercpp native libraries not found for '
                         f'{", ".join(missing)} (site-packages: {site}; collected: {names}). '
                         f'The frozen add-on would fail at first transcription.')
    print(f'bundling {len(found)} whisper.cpp native library file(s): {", ".join(sorted(names))}')
    return found


binaries = _whispercpp_binaries()

# pywhispercpp/model.py runs `importlib.metadata.version('pywhispercpp')` at
# import time. Without the dist-info, `import pywhispercpp.model` raises
# PackageNotFoundError and the add-on answers every host with `hello_error`.
# PyInstaller currently finds that call by bytecode scan; do not rely on it.
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata('pywhispercpp')

a = Analysis(
    [str(SPEC_ROOT / 'whispercpp_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=binaries,
    datas=datas,
    # `_pywhispercpp` is imported by pywhispercpp.model; listed explicitly so a
    # future lazy import upstream cannot silently drop it.
    hiddenimports=['_pywhispercpp', 'pywhispercpp.model', 'pywhispercpp.utils',
                   'pywhispercpp.constants', 'platformdirs', 'certifi'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'matplotlib', 'scipy', 'pandas', 'IPython', 'ipywidgets',
        'numpy.f2py', 'numpy.distutils', 'numpy.testing',
        'setuptools', 'pkg_resources', 'unittest', 'pydoc_data',
        # pywhispercpp's optional example/GUI extras.
        'sounddevice', 'webrtcvad',
        # NOTE: numpy, requests, tqdm and certifi must NOT be excluded — see
        # _REQUIRED above.
    ],
    noarchive=False,
)


# ---------------------------------------------------------------------------
# Drop the unmangled duplicates should anything drag them in (Linux only —
# on macOS a top-level entry with the same name can be PyInstaller's own
# symlink to the relocated copy).
# ---------------------------------------------------------------------------
def _drop_unmangled_duplicates(entries):
    if not sys.platform.startswith('linux'):
        return entries
    from importlib.metadata import distribution
    site = Path(distribution('pywhispercpp').locate_file('')).resolve()
    kept, dropped = [], 0
    for entry in entries:
        src = entry[1] if len(entry) > 1 else ''
        try:
            src_path = Path(src).resolve()
        except (OSError, TypeError):
            kept.append(entry)
            continue
        if src_path.parent == site and _UNMANGLED_ROOT_LIB.match(src_path.name):
            dropped += 1
            continue
        kept.append(entry)
    if dropped:
        print(f'dropped {dropped} unmangled whisper.cpp duplicate(s)')
    return kept


a.binaries = _drop_unmangled_duplicates(a.binaries)


# ---------------------------------------------------------------------------
# libexpat sanity check (verbatim from addon-sanotts).
#
# Do NOT blanket-drop the bundled libexpat the way a stdlib-only add-on can.
# With numpy in the graph, PyInstaller bundles whatever expat it resolved on
# the build host, and if that host has an OLD expat shadowing the distro one
# (e.g. a hand-built 2.5.0 in /usr/local/lib), the frozen binary dies before
# reaching Python:
#     Failed to load libpython3.x.so: undefined symbol: XML_SetReparseDeferralEnabled
# Dropping it is not a fix either — then the same stale system copy gets used.
#
# So verify the bundled copy actually exports the symbol libpython needs, and
# substitute a known-good system copy if it does not. On a clean CI runner this
# is a silent no-op; on a contaminated build host it turns a dead binary into a
# working one (or, failing that, a loud error instead of a mystery).
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so'):
            out.append(entry)
            continue
        if _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        replacement = None
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1',
                     '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                replacement = cand
                break
        if replacement:
            print(f'WARNING: build host libexpat at {src} lacks {_EXPAT_SYMBOL}; '
                  f'substituting {replacement}')
            out.append((name, replacement) + tuple(entry[2:]))
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL} and no good '
                  f'system copy was found. The frozen binary will not start. '
                  f'Fix the build host (a stale expat in /usr/local/lib is the '
                  f'usual cause).')
            out.append(entry)
    return out


a.binaries = _fix_libexpat(a.binaries)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whispercpp-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the OpenBLAS/ggml libraries saves little and
    # breaks macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='whispercpp-addon',
)
