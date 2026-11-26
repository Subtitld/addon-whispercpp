"""Manifest <-> code guards.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The host reads `display_name`, `languages`, `config_schema`, `tasks`,
`executable`, `protocol`, `startup_timeout_sec` and `platforms` straight from
manifest.json, never from the running add-on. Every test here exists because
some add-on shipped a manifest that silently disagreed with its code.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import whispercpp_addon  # noqa: E402
from whispercpp_addon import __main__ as addon  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
FIELDS = {f['key']: f for f in MANIFEST['config_schema']['fields']}
SOURCE = (REPO_ROOT / 'whispercpp_addon' / '__main__.py').read_text(encoding='utf-8')
PYPROJECT = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
WORKFLOW = (REPO_ROOT / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8')
SPEC = (REPO_ROOT / 'pyinstaller.spec').read_text(encoding='utf-8')
HARVESTER = (REPO_ROOT / 'tools' / 'fetch_third_party_licenses.py').read_text(encoding='utf-8')
README = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
LICENCES = REPO_ROOT / 'THIRD_PARTY_LICENSES'


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUADMATH_TOOL = _load_tool('build_libquadmath_source')


def test_identity_is_the_former_builtin():
    """Existing users' config and multi-GB model files are keyed by this id,
    and the host derives the WHISPERCPP_ env prefix from it."""
    assert MANIFEST['id'] == 'whispercpp' == addon.ADDON_ID == whispercpp_addon.ADDON_ID
    assert MANIFEST['display_name'] == 'Whisper (offline)'
    assert MANIFEST['tasks'] == ['asr.transcribe']
    assert MANIFEST['protocol'] == 1 == addon.PROTOCOL_VERSION
    assert MANIFEST['manifest_version'] == 1


def test_all_version_sites_agree():
    pyproject_version = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT, re.M).group(1)
    versions = {
        'manifest.json': MANIFEST['version'],
        'pyproject.toml': pyproject_version,
        '__init__.__version__': whispercpp_addon.__version__,
        '__main__.ADDON_VERSION': addon.ADDON_VERSION,
    }
    assert len(set(versions.values())) == 1, versions


def test_languages_match_code():
    assert MANIFEST['languages'] == list(addon._WHISPER_LANGUAGES)
    assert len(MANIFEST['languages']) == 99
    assert len(set(MANIFEST['languages'])) == 99


def test_model_ids_match_code():
    """Model ids are part of the on-disk filename; drift orphans downloads."""
    model = FIELDS['model']
    assert model['type'] == 'select'
    assert model['options'] == list(addon._AVAILABLE_MODELS)
    assert model['default'] == addon._DEFAULT_MODEL == 'base'
    assert [m['id'] for m in MANIFEST['models']] == list(addon._AVAILABLE_MODELS)
    assert all(m['file'] == f"ggml-{m['id']}.bin" for m in MANIFEST['models'])
    assert [m['id'] for m in MANIFEST['models'] if m.get('default')] == ['base']


def test_model_source_is_unchanged_from_the_builtin():
    assert addon._MODEL_BASE_URL == 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main'
    assert addon._model_filename('large-v3') == 'ggml-large-v3.bin'
    assert addon._USER_AGENT == 'subtitld-whispercpp/1'


def test_every_config_field_is_actually_read():
    """Both addon-piper-tts and addon-supertonic shipped config fields whose
    env var never matched what the code read — silently dead Configure
    dialogs. Assert every declared key has a corresponding WHISPERCPP_<KEY> read."""
    prefix = MANIFEST['id'].replace('-', '_').upper() + '_'
    for field in MANIFEST['config_schema']['fields']:
        env_name = prefix + field['key'].upper()
        assert f"'{env_name}'" in SOURCE, (
            f"config field {field['key']!r} declares {env_name}, "
            f'but nothing in __main__.py reads it')


def test_config_fields_use_renderable_types():
    """The Configure dialog renders bool/select/file (+ int inline only);
    anything else silently degrades to a text box."""
    allowed = {'bool', 'select', 'int', 'file', 'string', 'secret', 'password'}
    for field in MANIFEST['config_schema']['fields']:
        assert field['type'] in allowed, field


def test_no_int_fields_because_the_dialog_cannot_render_them():
    """`_AddonConfigDialog._render_field` has no `int` branch."""
    assert all(f['type'] != 'int' for f in MANIFEST['config_schema']['fields'])


def test_help_key_is_help_not_description():
    """`description` is ignored by the renderer; only `help` shows."""
    for field in MANIFEST['config_schema']['fields']:
        assert 'description' not in field, field['key']
        assert field.get('help'), f"{field['key']} has no help text"
        assert field.get('label'), f"{field['key']} has no label"


def test_select_defaults_are_among_their_options():
    for field in MANIFEST['config_schema']['fields']:
        if field['type'] == 'select':
            assert field['default'] in field['options'], field['key']


def test_threads_is_a_select_of_numeric_strings():
    threads = FIELDS['threads']
    assert threads['type'] == 'select'
    assert all(isinstance(o, str) for o in threads['options'])
    assert [int(o) for o in threads['options']] == sorted(int(o) for o in threads['options'])
    assert threads['default'] == '0'
    for option in threads['options']:
        assert addon._resolve_threads({'threads': option}) == int(option)


def test_config_keys_are_exactly_the_planned_ones():
    assert list(FIELDS) == ['model', 'threads', 'phrase_grouping', 'models_dir']
    assert FIELDS['phrase_grouping']['type'] == 'bool'
    assert FIELDS['phrase_grouping']['default'] is True
    assert FIELDS['models_dir']['type'] == 'file'


def test_pywhispercpp_is_pinned_exactly():
    pins = re.findall(r'"pywhispercpp==([^"]+)"', PYPROJECT)
    assert len(pins) == 1, 'pywhispercpp must be pinned with == exactly once'
    try:
        from importlib.metadata import version
        installed = version('pywhispercpp')
    except Exception:
        pytest.skip('pywhispercpp not installed')
    assert installed == pins[0], f'installed {installed}, pinned {pins[0]}'


def test_executable_matches_the_spec_and_the_staging_layout():
    assert MANIFEST['executable'] == 'bin/whispercpp-addon'
    assert re.search(r"name='whispercpp-addon'", SPEC)
    assert 'release-staging/bin' in WORKFLOW


def test_every_advertised_platform_is_built_and_tested():
    labels = set(re.findall(r'platform_label:\s*(\S+)', WORKFLOW))
    built = {label.split('-')[0] for label in labels}
    # Haiku's zip is built on Haiku (tools/build_haiku.py), not by CI.
    assert set(MANIFEST['platforms']) - {'haiku'} == built, (MANIFEST['platforms'], labels)
    assert (REPO_ROOT / 'tools' / 'build_haiku.py').is_file()
    # ...and the frozen binary is exercised by a real transcription.
    assert 'WHISPERCPP_TEST_BINARY' in WORKFLOW
    assert 'WHISPERCPP_TEST_TRANSCRIPTION' in WORKFLOW


def test_license_is_apache():
    assert MANIFEST['license'] == 'Apache-2.0'
    text = (REPO_ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert text.lstrip().startswith('Apache License')
    assert 'Version 2.0, January 2004' in text


def test_static_third_party_licences_are_present():
    root = LICENCES
    for rel in ('whisper.cpp/LICENSE', 'openai-whisper/LICENSE',
                'gcc-runtime/COPYING3', 'gcc-runtime/COPYING.RUNTIME',
                'libquadmath/COPYING.LIB', 'libquadmath/SOURCE.txt',
                'pybind11/LICENSE', 'pybind11/SOURCE.txt',
                'msvc-runtime/NOTICE.txt', 'python-runtime-libs/NOTICE.txt',
                'python-runtime-libs/openssl-LICENSE.txt'):
        assert (root / rel).is_file(), rel
    assert 'MIT License' in (root / 'whisper.cpp' / 'LICENSE').read_text(encoding='utf-8')
    lgpl = (root / 'libquadmath' / 'COPYING.LIB').read_text(encoding='utf-8')
    assert 'GNU LESSER GENERAL PUBLIC LICENSE' in lgpl and 'Version 2.1, February 1999' in lgpl
    pybind11 = (root / 'pybind11' / 'LICENSE').read_text(encoding='utf-8')
    assert 'Copyright (c) 2016 Wenzel Jakob' in pybind11
    assert 'Redistributions in binary form must reproduce' in pybind11


def test_libquadmath_is_not_labelled_gpl_with_the_runtime_exception():
    """libquadmath is LGPL-2.1; the GCC Runtime Library Exception covers
    libgcc/libstdc++/libgomp/libgfortran, not it."""
    gcc = (LICENCES / 'gcc-runtime' / 'SOURCE.txt').read_text(encoding='utf-8')
    for line in gcc.splitlines():
        if line.lstrip().startswith('*'):
            assert 'libquadmath' not in line, line
    assert 'LGPL' in gcc and '../libquadmath/' in gcc


def test_libquadmath_notice_matches_the_source_the_release_attaches():
    notice = (LICENCES / 'libquadmath' / 'SOURCE.txt').read_text(encoding='utf-8')
    tool = QUADMATH_TOOL
    for value in (tool.LIBQUADMATH_BUILD_ID, tool.LIBQUADMATH_SHA256, tool.SRPM_NAME,
                  tool.SRPM_SHA256, tool.SRPM_URLS[0], tool.GCC_TREE,
                  f'libquadmath-{tool.RPM_RELEASE}', 'LGPL-2.1-or-later',
                  'whispercpp-addon-<version>-libquadmath-source.tar.gz'):
        assert value in notice, value
    assert MANIFEST['homepage'] + '/releases' in notice


def test_every_platform_specific_licence_is_gated_and_required():
    only = re.search(r'_PLATFORM_ONLY = (\{[^}]*\})', HARVESTER).group(1)
    assert "'libquadmath': 'linux'" in only
    assert "'gcc-runtime': 'linux'" in only
    for pattern in (r"r'^pybind11/LICENSE$'", r"r'^libquadmath/COPYING\.LIB$'",
                    r"r'^libquadmath/SOURCE\.txt$'"):
        assert pattern in HARVESTER, pattern
    for needle in (r'THIRD_PARTY_LICENSES/pybind11/LICENSE',
                   r'THIRD_PARTY_LICENSES/libquadmath/COPYING\.LIB',
                   r'THIRD_PARTY_LICENSES/libquadmath/SOURCE\.txt'):
        assert needle in WORKFLOW, needle


def test_release_attaches_the_libquadmath_source_and_checks_the_bundle():
    assert re.search(r'tools/build_libquadmath_source\.py [^\n]*--out', WORKFLOW)
    assert 'tools/build_libquadmath_source.py --check-bundle' in WORKFLOW
    assert 'libquadmath-source.tar.gz' in WORKFLOW
    # The asset must not look like a platform zip to the catalog builder.
    assert not re.search(r'libquadmath-source[^\s]*\.zip', WORKFLOW)


def test_frozen_download_step_hides_the_runner_certificates():
    """The runners have CA certificates where the bundled OpenSSL looks for
    them; users' machines often do not. The frozen suite must not lean on them."""
    step = WORKFLOW.split('Prove the frozen binary actually transcribes', 1)[1].split('- name:', 1)[0]
    assert re.search(r"SSL_CERT_FILE: \$\{\{ runner\.temp \}\}/no-such-ca\.pem", step)
    assert re.search(r"SSL_CERT_DIR: \$\{\{ runner\.temp \}\}/no-such-ca-dir", step)


def test_certifi_is_a_declared_runtime_dependency():
    assert re.search(r'^\s*"certifi[^"]*",', PYPROJECT, re.M)
    assert "'certifi'" in SPEC


def test_the_libquadmath_numpy_ships_is_the_one_the_release_has_source_for():
    """A numpy/OpenBLAS upgrade can swap libquadmath for another build; the
    attached source would then no longer correspond. Caught here, before a
    frozen build, wherever numpy vendors it (Linux)."""
    try:
        import numpy
    except ImportError:
        pytest.skip('numpy not installed')
    libs = Path(numpy.__file__).resolve().parent.parent / 'numpy.libs'
    found = sorted(libs.glob('libquadmath*.so*')) if libs.is_dir() else []
    if sys.platform.startswith('linux'):
        assert found, f'numpy no longer vendors libquadmath in {libs}: drop the LGPL notice?'
    for path in found:
        assert QUADMATH_TOOL.elf_build_id(path) == QUADMATH_TOOL.LIBQUADMATH_BUILD_ID, path


def test_readme_does_not_overclaim_licence_coverage():
    assert 'covers everything frozen' not in README
    assert 'LGPL' in README and 'libquadmath' in README and 'pybind11' in README
    body = WORKFLOW.split('body: |', 1)[1]
    assert 'numpy and its OpenBLAS are BSD-3-Clause' not in body
    assert 'libquadmath' in body and 'LGPL' in body


def test_libquadmath_tarball_keeps_what_its_sources_include():
    """The attached source must build: printf/gmp-impl.h includes
    ../../libgcc/longlong.h, whose macros are compiled into the library."""
    tool = QUADMATH_TOOL
    assert tool._keep_member('libgcc/longlong.h', False)
    kept = {'libquadmath/printf/gmp-impl.h', 'libgcc/longlong.h', 'libquadmath/quadmath.h'}
    sources = {
        'libquadmath/printf/gmp-impl.h': b'#include "../../libgcc/longlong.h"\n#include <stdlib.h>\n',
        'libquadmath/printf/mul_1.c': b'#  include "../quadmath.h"\n#include "gmp-impl.h"\n',
    }
    assert tool.missing_includes(sources, kept) == []
    sources['libquadmath/math/x.c'] = b'#include "../../gcc/tm.h"\n'
    assert tool.missing_includes(sources, kept) == ['libquadmath/math/x.c -> gcc/tm.h']
