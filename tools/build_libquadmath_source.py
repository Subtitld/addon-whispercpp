#!/usr/bin/env python3
"""Build the libquadmath source tarball attached to every release.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Why: the Linux archive ships libquadmath (LGPL-2.1-or-later), vendored by
numpy's OpenBLAS. LGPL-2.1 §4 lets object code be distributed by offering
the complete corresponding source "from a designated place" alongside it,
with no written-offer shortcut. The release job attaches this tarball to the
same GitHub release as the binaries.

Which source: the shipped file is the unmodified libquadmath.so.0.0.0 of
CentOS 7's libquadmath-4.8.5-44.el7 (the OpenBLAS wheel's manylinux2014 build
image), only renamed by auditwheel. Its GNU build-id and the sha256 prefix in
its mangled name both match that package. So the source is Red Hat's source
package gcc-4.8.5-44.el7.src.rpm, not the upstream gcc-4.8.5 tarball
(Red Hat builds a 2015-07-02 snapshot of the 4.8 branch). From it the tarball
keeps libquadmath/, the top-level build machinery its configure and Makefile
use, and Red Hat's spec and patches (~2 MB) — not the other ~70 MB of the
compiler. The full package stays linked in the README.

    python tools/build_libquadmath_source.py --out whispercpp-addon-0.0.1-libquadmath-source.tar.gz
    python tools/build_libquadmath_source.py --check-bundle dist/whispercpp-addon

`--check-bundle` fails when a bundled libquadmath is NOT the build this
tarball corresponds to (a numpy / OpenBLAS upgrade that changes it must
update the constants below, the tarball and ../THIRD_PARTY_LICENSES/libquadmath/).
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import posixpath
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Must match THIRD_PARTY_LICENSES/libquadmath/SOURCE.txt (a test checks it).
RPM_RELEASE = '4.8.5-44.el7'
LIBQUADMATH_BUILD_ID = '549b4c82347785459571c79239872ad31509dcf4'
LIBQUADMATH_SHA256 = '96973f995bad4e4b80eaa188b7bac60bd0df44b22ee67bd046b3aa4ecb9e34fd'
SRPM_NAME = f'gcc-{RPM_RELEASE}.src.rpm'
SRPM_SHA256 = '642c4085f6af9565e39583e54a9d7a5d9e6b93dd8c8e9474aec8a892d6c7aa36'
SRPM_URLS = (
    f'https://vault.centos.org/7.9.2009/os/Source/SPackages/{SRPM_NAME}',
    f'https://archive.kernel.org/centos-vault/7.9.2009/os/Source/SPackages/{SRPM_NAME}',
)
GCC_TARBALL = 'gcc-4.8.5-20150702.tar.bz2'
GCC_TREE = 'gcc-4.8.5-20150702'
BINARY_RPM_URL = ('https://vault.centos.org/7.9.2009/os/x86_64/Packages/'
                  f'libquadmath-{RPM_RELEASE}.x86_64.rpm')

# Inside GCC_TREE: what building libquadmath reads besides libquadmath/
# (its configure sources ../config-ml.in, ../ltmain.sh, ../config/*.m4;
# its Makefile reads ../gcc/BASE-VER, ../contrib/make_sunver.pl and the
# texinfo includes; printf/gmp-impl.h includes ../../libgcc/longlong.h,
# whose macros are compiled into the library). build() refuses to write a
# tarball in which any kept source includes a `../` file left out.
_KEEP_DIRS = ('libquadmath/', 'config/', 'gcc/doc/include/')
_KEEP_FILES = ('gcc/BASE-VER', 'gcc/DATESTAMP', 'gcc/DEV-PHASE', 'contrib/make_sunver.pl',
               'libgcc/longlong.h')
_INCLUDE_RE = re.compile(rb'^\s*#\s*include\s+"(\.\./[^"]+)"', re.M)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# --check-bundle
# ---------------------------------------------------------------------------
def elf_build_id(path: Path) -> str | None:
    """The NT_GNU_BUILD_ID of a 64-bit little-endian ELF file, or None."""
    with open(path, 'rb') as f:
        ident = f.read(16)
        if ident[:4] != b'\x7fELF' or ident[4] != 2 or ident[5] != 1:
            return None
        header = f.read(48)
        e_phoff = struct.unpack_from('<Q', header, 16)[0]
        e_phentsize, e_phnum = struct.unpack_from('<HH', header, 38)
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            p_type, _flags, p_offset, _vaddr, _paddr, p_filesz = struct.unpack('<IIQQQQ', f.read(40))
            if p_type != 4:  # PT_NOTE
                continue
            f.seek(p_offset)
            notes = f.read(p_filesz)
            pos = 0
            while pos + 12 <= len(notes):
                namesz, descsz, ntype = struct.unpack_from('<III', notes, pos)
                pos += 12
                name = notes[pos:pos + namesz]
                pos += (namesz + 3) & ~3
                desc = notes[pos:pos + descsz]
                pos += (descsz + 3) & ~3
                if ntype == 3 and name.rstrip(b'\0') == b'GNU':  # NT_GNU_BUILD_ID
                    return desc.hex()
    return None


def check_bundle(root: Path) -> int:
    found = sorted(p for p in root.rglob('libquadmath*.so*')
                   if p.is_file() and not p.is_symlink())
    if not found:
        _log(f'no libquadmath in {root}; nothing to check')
        return 0
    bad = 0
    for path in found:
        build_id = elf_build_id(path)
        ok = build_id == LIBQUADMATH_BUILD_ID
        bad += not ok
        _log(f'{"ok " if ok else "BAD"} {path.relative_to(root)} build-id {build_id}')
    if bad:
        _log(f'\nFATAL: {bad} bundled libquadmath file(s) are not the libquadmath-{RPM_RELEASE} '
             f'build (build-id {LIBQUADMATH_BUILD_ID}) whose source the release attaches. '
             f'Find the new one\'s origin and update tools/build_libquadmath_source.py and '
             f'THIRD_PARTY_LICENSES/libquadmath/.')
        return 1
    return 0


# ---------------------------------------------------------------------------
# RPM / cpio reading (no rpm2cpio on the runner needed)
# ---------------------------------------------------------------------------
def _rpm_payload(raw) -> io.BufferedIOBase:
    """The decompressed cpio stream of an RPM file object."""
    if raw.read(4) != b'\xed\xab\xee\xdb':
        raise SystemExit('FATAL: not an RPM file')
    raw.seek(96)  # the lead
    for padded in (True, False):  # signature header, then the main header
        intro = raw.read(16)
        if intro[:3] != b'\x8e\xad\xe8':
            raise SystemExit('FATAL: malformed RPM header')
        nindex, hsize = struct.unpack('>II', intro[8:16])
        size = 16 * nindex + hsize
        raw.seek(size + ((8 - size % 8) % 8 if padded else 0), os.SEEK_CUR)
    start = raw.tell()
    magic = raw.read(6)
    raw.seek(start)
    if magic.startswith(b'\x1f\x8b'):
        return gzip.GzipFile(fileobj=raw)
    if magic.startswith(b'\xfd7zXZ'):
        return lzma.LZMAFile(raw)
    if magic.startswith(b'BZh'):
        return bz2.BZ2File(raw)
    raise SystemExit(f'FATAL: unsupported RPM payload compression {magic!r}')


def _read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise SystemExit('FATAL: truncated RPM payload')
    return data


def _extract_rpm(rpm: Path, dest: Path, wanted) -> list[str]:
    """Extract the regular files whose name satisfies `wanted` (newc cpio)."""
    extracted: list[str] = []
    with open(rpm, 'rb') as raw, _rpm_payload(raw) as stream:
        while True:
            header = _read_exact(stream, 110)
            if header[:6] not in (b'070701', b'070702'):
                raise SystemExit(f'FATAL: unexpected cpio header {header[:6]!r}')
            fields = [int(header[6 + 8 * i:14 + 8 * i], 16) for i in range(13)]
            mode, filesize, namesize = fields[1], fields[6], fields[11]
            name = _read_exact(stream, namesize)[:-1].decode('utf-8')
            _read_exact(stream, (4 - (110 + namesize) % 4) % 4)
            if name == 'TRAILER!!!':
                return extracted
            name = name.removeprefix('./')
            keep = (mode & 0o170000) == 0o100000 and wanted(name)
            target = dest / Path(name).name if keep else None
            out = open(target, 'wb') if target else None
            try:
                left = filesize
                while left:
                    chunk = stream.read(min(left, 1 << 20))
                    if not chunk:
                        raise SystemExit('FATAL: truncated RPM payload')
                    left -= len(chunk)
                    if out:
                        out.write(chunk)
            finally:
                if out:
                    out.close()
            if keep:
                extracted.append(Path(name).name)
            _read_exact(stream, (4 - filesize % 4) % 4)


# ---------------------------------------------------------------------------
# --out
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _fetch_srpm(dest: Path) -> Path:
    target = dest / SRPM_NAME
    for url in SRPM_URLS:
        _log(f'fetching {url}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'subtitld-addon-whispercpp'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(target, 'wb') as out:
                shutil.copyfileobj(resp, out, 1 << 20)
        except OSError as exc:
            _log(f'  failed: {exc}')
            continue
        digest = _sha256(target)
        if digest == SRPM_SHA256:
            return target
        _log(f'  sha256 {digest} does not match the pinned {SRPM_SHA256}')
    raise SystemExit(f'FATAL: could not fetch a verified {SRPM_NAME}')


def _keep_member(rel: str, is_dir: bool) -> bool:
    if '/' not in rel:
        return not is_dir  # top-level build machinery: configure, Makefile.in, ltmain.sh, ...
    return rel.startswith(_KEEP_DIRS) or rel in _KEEP_FILES


def missing_includes(sources: dict[str, bytes], kept: set[str]) -> list[str]:
    """`from -> to` for every `#include "../..."` in `sources` (tree-relative
    path -> content) that resolves to a file not in `kept`."""
    missing = []
    for rel, data in sorted(sources.items()):
        for target in _INCLUDE_RE.findall(data):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target.decode()))
            if resolved not in kept:
                missing.append(f'{rel} -> {resolved}')
    return missing


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def build(out: Path, srpm: Path | None, version: str) -> int:
    prefix = out.name.removesuffix('.tar.gz')
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if srpm is None:
            srpm = _fetch_srpm(work)
        elif _sha256(srpm) != SRPM_SHA256:
            raise SystemExit(f'FATAL: {srpm} is not the pinned {SRPM_NAME}')
        _log(f'extracting {SRPM_NAME}')
        files = _extract_rpm(srpm, work,
                             lambda n: n in (GCC_TARBALL, 'gcc.spec') or n.endswith('.patch'))
        patches = sorted(f for f in files if f.endswith('.patch'))
        if GCC_TARBALL not in files or 'gcc.spec' not in files or not patches:
            raise SystemExit(f'FATAL: {SRPM_NAME} lacks {GCC_TARBALL}, gcc.spec or its patches')
        touching = [p for p in patches
                    if re.search(rb'^(\+\+\+|---) \S*libquadmath/', (work / p).read_bytes(), re.M)]
        mtime = 1435795200  # 2015-07-02, the GCC snapshot date: reproducible output

        out.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        kept_names: set[str] = set()
        quadmath_sources: dict[str, bytes] = {}
        with open(out, 'wb') as raw, \
                gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz, \
                tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar, \
                tarfile.open(work / GCC_TARBALL, 'r:bz2') as src:
            for member in src:
                rel = member.name.removeprefix(GCC_TREE + '/')
                if not member.name.startswith(GCC_TREE + '/') or not rel:
                    continue
                if not _keep_member(rel, member.isdir()):
                    continue
                member.name = f'{prefix}/{GCC_TREE}/{rel}'
                if member.issym() or member.islnk():
                    continue  # none in these directories; never follow one out of the tree
                member.uid = member.gid = 0
                member.uname = member.gname = ''
                if member.isfile():
                    data = src.extractfile(member).read()
                    kept_names.add(rel)
                    if rel.startswith('libquadmath/') and rel.endswith(('.c', '.h')):
                        quadmath_sources[rel] = data
                    tar.addfile(member, io.BytesIO(data))
                    kept += 1
                else:
                    tar.addfile(member)
            for name in ['gcc.spec'] + patches:
                sub = 'redhat' if name == 'gcc.spec' else 'redhat/patches'
                _add_bytes(tar, f'{prefix}/{sub}/{name}', (work / name).read_bytes(), mtime)
            readme = _README.format(
                version=version, rpm=RPM_RELEASE, srpm=SRPM_NAME, srpm_sha=SRPM_SHA256,
                srpm_url=SRPM_URLS[0], binary_rpm=BINARY_RPM_URL, build_id=LIBQUADMATH_BUILD_ID,
                lib_sha=LIBQUADMATH_SHA256, tree=GCC_TREE, tarball=GCC_TARBALL,
                patch_count=len(patches),
                touching=', '.join(touching) if touching else 'none of them',
                kept=kept)
            _add_bytes(tar, f'{prefix}/README.md', readme.encode('utf-8'), mtime)
        missing = missing_includes(quadmath_sources, kept_names)
        if missing:
            out.unlink()
            raise SystemExit('FATAL: the tarball would not build libquadmath; these includes '
                             'point at files left out (add them to _KEEP_FILES):\n  '
                             + '\n  '.join(missing))
    _log(f'wrote {out} ({out.stat().st_size / 2**20:.1f} MiB, {kept} files from {GCC_TREE}, '
         f'{len(patches)} patches; patches touching libquadmath/: {touching or "none"})')
    return 0


_README = """# libquadmath source for the Subtitld whisper.cpp add-on {version}

The Linux archive of this add-on release ships libquadmath, licensed under
the GNU Lesser General Public License 2.1 or later
(`THIRD_PARTY_LICENSES/libquadmath/COPYING.LIB` in the archive). This
tarball is its complete corresponding source, offered from the same place
as the binaries (LGPL-2.1 section 4).

## Which build

`bin/_internal/numpy.libs/libquadmath-96973f99-*.so.0.0.0` (and its copy in
`bin/_internal/`) comes with numpy's OpenBLAS. It is the unmodified
`/usr/lib64/libquadmath.so.0.0.0` of CentOS 7's `libquadmath-{rpm}.x86_64`
({binary_rpm}), with only its SONAME renamed by auditwheel:

| | |
|---|---|
| GNU build-id | `{build_id}` |
| sha256 of the original file | `{lib_sha}` |

The release build refuses to publish when the bundled file's build-id
differs.

## Contents

| Path | What |
|---|---|
| `{tree}/libquadmath/` | the library's source |
| `{tree}/*`, `config/`, `gcc/BASE-VER` ..., `contrib/make_sunver.pl`, `gcc/doc/include/` | the GCC build machinery that configures and builds it |
| `{tree}/libgcc/longlong.h` | a header the library's printf code includes |
| `redhat/gcc.spec` | Red Hat's build recipe (see `%build`; libquadmath is built as a GCC target library) |
| `redhat/patches/` | all {patch_count} patches of the source package; {touching} touch `libquadmath/` |

The files come from `{tarball}` inside Red Hat's source package
`{srpm}`:

    {srpm_url}
    sha256 {srpm_sha}

That package also holds the rest of the compiler, which is not part of the
library and is left out here ({kept} files kept). Upstream GCC 4.8.5 is at
https://ftp.gnu.org/gnu/gcc/gcc-4.8.5/.

## Rebuilding

None of Red Hat's patches change the files kept here, so nothing needs
applying. libquadmath configures and builds on its own, out of tree, with
the system C compiler:

    mkdir build && cd build
    ../{tree}/libquadmath/configure --disable-multilib
    make

The library is `build/.libs/libquadmath.so.0.0.0`; copy it over the shipped
file (keeping the shipped file name). Red Hat built it as a target library
of a full GCC build (`redhat/gcc.spec`, `%build`); for that, use the whole
source package above.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='tarball to write (*.tar.gz)')
    mode.add_argument('--check-bundle', type=Path, metavar='DIR',
                      help='verify every libquadmath under DIR is the pinned build')
    ap.add_argument('--srpm', type=Path, help=f'use a local {SRPM_NAME} instead of downloading')
    ap.add_argument('--version', help='add-on version for the README (default: manifest.json)')
    args = ap.parse_args(argv)
    if args.check_bundle:
        return check_bundle(args.check_bundle)
    if not args.out.name.endswith('.tar.gz'):
        ap.error('--out must end in .tar.gz')
    version = args.version or json.loads(
        (Path(__file__).resolve().parent.parent / 'manifest.json').read_text(encoding='utf-8'))['version']
    return build(args.out, args.srpm, version)


if __name__ == '__main__':
    raise SystemExit(main())
