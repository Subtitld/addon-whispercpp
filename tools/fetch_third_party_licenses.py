#!/usr/bin/env python3
"""Assemble THIRD_PARTY_LICENSES/ for the distributed archive.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The frozen bundle carries CPython and its native libraries (OpenSSL, expat,
libffi, zlib, ...), the PyInstaller bootloader, pywhispercpp, the
whisper.cpp/ggml native libraries, numpy (with OpenBLAS and the GCC Fortran
runtime), requests and friends, and — per platform — the GCC runtime
(libgomp, libstdc++) or the MSVC runtime. PyInstaller drops `.dist-info`
licence files (except the ones it needs for metadata), so the notices have to
be put back explicitly.

Two sources:

  * **Harvested** from the build environment: every installed distribution in
    pywhispercpp's and platformdirs' runtime dependency closure, with nested
    paths preserved (numpy ships several files all called LICENSE.md), plus
    CPython's own licence and PyInstaller's (bootloader exception).
  * **Committed** in this repository's THIRD_PARTY_LICENSES/: texts the wheels
    do not carry — whisper.cpp, OpenAI Whisper (the model weights), the GCC
    runtime's GPL + Runtime Library Exception (Linux), the MSVC runtime
    notice (Windows) and the Python runtime's native libraries. `--refresh`
    re-fetches the upstream ones from pinned URLs.

This is a BUILD-FAILING step: if a required notice is missing, the archive
must not ship.

    python tools/fetch_third_party_licenses.py --out release-staging/THIRD_PARTY_LICENSES
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import sysconfig
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / 'THIRD_PARTY_LICENSES'

# Distributions whose runtime closure is frozen into the binary.
ROOTS = ('pywhispercpp', 'platformdirs')

WHISPER_CPP_TAG = 'v1.8.5'  # the version inside pywhispercpp 1.5.1
# pywhispercpp v1.5.1's pybind11 submodule (3.0.0.dev1).
PYBIND11_COMMIT = 'b70b8eb332fadf55d7e22b492da0e954c1a4fcb7'
OPENAI_WHISPER_TAG = 'v20250625'
_REFRESH = {
    'whisper.cpp/LICENSE':
        f'https://raw.githubusercontent.com/ggml-org/whisper.cpp/{WHISPER_CPP_TAG}/LICENSE',
    'pybind11/LICENSE':
        f'https://raw.githubusercontent.com/pybind/pybind11/{PYBIND11_COMMIT}/LICENSE',
    'openai-whisper/LICENSE':
        f'https://raw.githubusercontent.com/openai/whisper/{OPENAI_WHISPER_TAG}/LICENSE',
    'gcc-runtime/COPYING3':
        'https://raw.githubusercontent.com/gcc-mirror/gcc/master/COPYING3',
    'gcc-runtime/COPYING.RUNTIME':
        'https://raw.githubusercontent.com/gcc-mirror/gcc/master/COPYING.RUNTIME',
    'python-runtime-libs/openssl-LICENSE.txt':
        'https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.13/LICENSE.txt',
    'python-runtime-libs/expat-COPYING':
        'https://raw.githubusercontent.com/libexpat/libexpat/R_2_6_1/expat/COPYING',
    'python-runtime-libs/libffi-LICENSE':
        'https://raw.githubusercontent.com/libffi/libffi/v3.4.6/LICENSE',
    'python-runtime-libs/zlib-LICENSE':
        'https://raw.githubusercontent.com/madler/zlib/v1.3.1/LICENSE',
    'python-runtime-libs/bzip2-LICENSE':
        'https://gitlab.com/bzip2/bzip2/-/raw/bzip2-1.0.8/LICENSE',
    'python-runtime-libs/xz-COPYING':
        'https://raw.githubusercontent.com/tukaani-project/xz/v5.4.5/COPYING',
}

# Static components that only apply to one platform's bundle.
_PLATFORM_ONLY = {'gcc-runtime': 'linux', 'msvc-runtime': 'win32'}

_LICENSE_NAME = re.compile(r'^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)', re.I)


def _norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _platform_key() -> str:
    if sys.platform.startswith('linux'):
        return 'linux'
    return sys.platform


def _requirement_name(req: str) -> str | None:
    """Name of a runtime requirement, or None for extras-only ones."""
    spec, _, marker = req.partition(';')
    if 'extra' in marker:
        return None
    match = re.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', spec)
    return match.group(1) if match else None


def _closure(roots) -> dict:
    from importlib.metadata import PackageNotFoundError, distribution

    seen: dict = {}
    todo = list(roots)
    while todo:
        name = todo.pop()
        key = _norm(name)
        if key in seen:
            continue
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue  # a marker-excluded dependency that is not installed
        seen[key] = dist
        for req in dist.requires or []:
            dep = _requirement_name(req)
            if dep:
                todo.append(dep)
    return seen


def _license_field(dist) -> str:
    meta = dist.metadata
    value = meta.get('License-Expression') or ''
    if not value:
        raw = (meta.get('License') or '').strip()
        if raw and len(raw) < 80 and '\n' not in raw:
            value = raw
    if not value:
        classifiers = [c.split('::')[-1].strip() for c in meta.get_all('Classifier') or []
                       if c.startswith('License ::')]
        value = ', '.join(classifiers)
    return value or 'see licence files'


def _harvest_dist(dist, out: Path) -> list[str]:
    name, version = dist.metadata['Name'], dist.version
    target = out / f'{name}-{version}'
    copied: list[str] = []
    for rel in dist.files or []:
        parts = Path(str(rel)).parts
        if not parts or not parts[0].endswith('.dist-info'):
            continue
        if len(parts) > 2 and parts[1] == 'licenses':
            inner = Path(*parts[2:])
        elif len(parts) == 2 and _LICENSE_NAME.match(parts[1]):
            inner = Path(parts[1])
        else:
            continue
        src = Path(dist.locate_file(rel))
        if not src.is_file():
            continue
        dest = target / inner
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied.append(str(dest.relative_to(out)).replace('\\', '/'))
    if copied:
        (target / 'SOURCE.txt').write_text(
            f'{name} {version}\nLicence: {_license_field(dist)}\n'
            f'Harvested from the installed distribution used for this build.\n',
            encoding='utf-8')
    return copied


def _harvest_python(out: Path) -> list[str]:
    version = sys.version.split()[0]
    candidates = [Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt',
                  Path(sys.base_prefix) / 'LICENSE.txt',
                  Path(sys.base_prefix) / 'LICENSE']
    for src in candidates:
        if src.is_file():
            target = out / f'python-{version}'
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target / 'LICENSE.txt')
            (target / 'SOURCE.txt').write_text(
                f'CPython {version}\nLicence: PSF-2.0 (see LICENSE.txt)\n'
                f'The interpreter runtime frozen into bin/ by PyInstaller.\n', encoding='utf-8')
            return [f'python-{version}/LICENSE.txt']
    print(f'WARNING: no CPython licence found (looked in {candidates})', file=sys.stderr)
    return []


def _copy_static(out: Path) -> list[str]:
    copied: list[str] = []
    for component in sorted(p for p in STATIC_DIR.iterdir() if p.is_dir()):
        only = _PLATFORM_ONLY.get(component.name)
        if only and only != _platform_key():
            continue
        for src in sorted(component.rglob('*')):
            if src.is_file():
                dest = out / src.relative_to(STATIC_DIR)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                copied.append(str(dest.relative_to(out)).replace('\\', '/'))
    return copied


def _refresh() -> int:
    for rel, url in _REFRESH.items():
        req = urllib.request.Request(url, headers={'User-Agent': 'subtitld-addon-whispercpp'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if b'<html' in data[:500].lower():
            print(f'FAILED {rel}: got HTML from {url}', file=sys.stderr)
            return 1
        dest = STATIC_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f'refreshed {rel} ({len(data)} bytes)', file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, help='directory to assemble into')
    ap.add_argument('--refresh', action='store_true',
                    help=f're-fetch the committed upstream texts into {STATIC_DIR.name}/')
    args = ap.parse_args(argv)
    if args.refresh:
        return _refresh()
    if args.out is None:
        ap.error('--out is required')

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows: list[tuple[str, str, list[str]]] = []
    for key, dist in sorted(_closure(ROOTS).items()):
        files = _harvest_dist(dist, out)
        rows.append((f"{dist.metadata['Name']} {dist.version}", _license_field(dist), files))
    try:
        from importlib.metadata import distribution
        pyi = distribution('pyinstaller')
        files = _harvest_dist(pyi, out)
        rows.append((f'PyInstaller {pyi.version} (bootloader)',
                     'GPL-2.0-or-later WITH Bootloader-exception', files))
    except Exception:  # noqa: BLE001
        print('WARNING: PyInstaller is not installed; its bootloader notice is missing',
              file=sys.stderr)
    rows.append((f'CPython {sys.version.split()[0]}', 'PSF-2.0', _harvest_python(out)))
    static = _copy_static(out)

    lines = [
        '# Third-party licences',
        '',
        'The Subtitld whisper.cpp add-on itself is licensed under the Apache License 2.0',
        '(see ../LICENSE). This archive also contains the third-party components below,',
        'each under its own licence. Each directory holds the licence text(s) and a',
        'SOURCE.txt / NOTICE.txt naming the component and version.',
        '',
        'Speech models are NOT included: the GGML model files download on first use',
        'from https://huggingface.co/ggerganov/whisper.cpp (MIT, see openai-whisper/).',
        '',
        '## Bundled Python distributions and runtime',
        '',
    ]
    for label, licence, files in rows:
        shown = ', '.join(files) if files else '(no licence file shipped)'
        lines.append(f'- **{label}** — {licence} — {shown}')
    lines += ['', '## Native components and model weights', '']
    for component in sorted({f.split('/', 1)[0] for f in static}):
        names = sorted(f.split('/', 1)[1] for f in static if f.startswith(component + '/'))
        lines.append(f'- **{component}** — {", ".join(names)}')
    lines.append('')
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')

    present = {str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()}
    required = [r'^pywhispercpp-[^/]+/LICENSE$', r'^numpy-[^/]+/LICENSE\.txt$',
                r'^whisper\.cpp/LICENSE$', r'^pybind11/LICENSE$', r'^openai-whisper/LICENSE$',
                r'^python-[^/]+/LICENSE\.txt$', r'^python-runtime-libs/openssl-LICENSE\.txt$',
                r'^README\.md$']
    if _platform_key() == 'linux':
        required += [r'^gcc-runtime/COPYING3$', r'^gcc-runtime/COPYING\.RUNTIME$']
    if _platform_key() == 'win32':
        required += [r'^msvc-runtime/NOTICE\.txt$']
    missing = [r for r in required if not any(re.match(r, p) for p in present)]
    if missing:
        print('\nFATAL: required licence files missing:', file=sys.stderr)
        for pattern in missing:
            print(f'  {pattern}', file=sys.stderr)
        return 1
    print(f'THIRD_PARTY_LICENSES assembled at {out}: {len(present)} file(s), '
          f'{len(rows)} bundled distribution(s)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
