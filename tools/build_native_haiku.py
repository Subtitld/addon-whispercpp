#!/usr/bin/env python3
"""Build pywhispercpp on Haiku and lay it out in build/haiku-lib/ for tools/build_haiku.py.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python3.14 tools/build_native_haiku.py [--jobs N]

PyPI has no pywhispercpp wheel for Haiku, and HaikuPorts' Python has no pip,
so this builds the pinned sdist with CMake directly, the way its setup.py
does, and writes what the wheel would hold:

- build/haiku-lib/_pywhispercpp<suffix>.so: the extension, with whisper.cpp
  and ggml linked in statically (one file; nothing to find at run time). Its
  CPU code targets x86-64-v3 (AVX2, FMA, F16C), as ggml's static builds do.
- build/haiku-lib/pywhispercpp/: the Python package, and a dist-info with its
  version, which pywhispercpp reads through importlib.metadata.
- build/haiku-lib/tqdm/: tqdm, a pure-Python wheel pywhispercpp imports,
  checked by sha256.

numpy, requests, platformdirs and certifi come from Haiku packages (the
manifest's haiku_packages).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
SRC = BUILD / 'haiku-src'
NATIVE_BUILD = BUILD / 'haiku-native'
OUT = BUILD / 'haiku-lib'
DOWNLOADS = BUILD / 'downloads'

PYWHISPERCPP_VERSION = '1.5.1'
SDIST = ('https://files.pythonhosted.org/packages/9f/35/610787a277e177e6b7983277a35a2969c0876b46c0d889bc6a0c4b4ac4a7/'
         'pywhispercpp-1.5.1.tar.gz', '5df897e5dd9d9f16804fc937bd85b9f5880a0b29a55f8843585e0e03d76195e5')
TQDM = ('https://files.pythonhosted.org/packages/a7/03/921a3d3c75785aca9ebfbfcabfbc3a1be12e2ab5265deb026d55a5a3f83e/'
        'tqdm-4.70.1-py3-none-any.whl', 'c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73')

CMAKE_OPTIONS = [
    '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=OFF', '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    '-DPYBIND11_FINDPYTHON=ON',
    '-DWHISPER_BUILD_TESTS=OFF', '-DWHISPER_BUILD_EXAMPLES=OFF', '-DWHISPER_BUILD_SERVER=OFF',
    '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
    '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
]


def run(cmd, cwd=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, expected: str) -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / url.rsplit('/', 1)[1]
    if not (target.is_file() and sha256(target) == expected):
        print('+ fetch', url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as out:
            shutil.copyfileobj(response, out)
    if sha256(target) != expected:
        sys.exit(f'FATAL: {target.name} does not match its pinned sha256')
    return target


def fresh(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> int:
    if not sys.platform.startswith('haiku'):
        sys.exit('FATAL: run this on Haiku')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()

    fresh(SRC)
    with tarfile.open(fetch(*SDIST)) as tf:
        tf.extractall(SRC, filter='data')
    source = SRC / f'pywhispercpp-{PYWHISPERCPP_VERSION}'

    fresh(OUT)
    fresh(NATIVE_BUILD)
    run(['cmake', '-S', source, '-B', NATIVE_BUILD, *CMAKE_OPTIONS,
         f'-DPython_EXECUTABLE={sys.executable}', f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={OUT}'])
    run(['cmake', '--build', NATIVE_BUILD, '--parallel', str(args.jobs), '--target', '_pywhispercpp'])

    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    extension = OUT / f'_pywhispercpp{suffix}'
    if not extension.is_file():
        sys.exit(f'FATAL: the build made no {extension.name} in {OUT}')
    run(['strip', '--strip-unneeded', extension])

    shutil.copytree(source / 'pywhispercpp', OUT / 'pywhispercpp',
                    ignore=shutil.ignore_patterns('examples', '__pycache__'))
    info = OUT / f'pywhispercpp-{PYWHISPERCPP_VERSION}.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: pywhispercpp\nVersion: {PYWHISPERCPP_VERSION}\n',
                                   encoding='utf-8')
    (info / 'INSTALLER').write_text('subtitld-addon-whispercpp tools/build_native_haiku.py\n', encoding='utf-8')

    with zipfile.ZipFile(fetch(*TQDM)) as zf:
        zf.extractall(OUT)

    sys.path.insert(0, str(OUT))
    import _pywhispercpp  # noqa: F401  — proof the extension loads here
    print(f'{extension.name}: {extension.stat().st_size / 2**20:.1f} MiB; loads')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
