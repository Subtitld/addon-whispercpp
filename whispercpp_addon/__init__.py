"""Subtitld whisper.cpp add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The package exists so the entry point can run both from source
(`python -m whispercpp_addon`) and as the frozen PyInstaller binary
(`bin/whispercpp-addon`).
"""

__version__ = '0.0.1'
# Kept from the former built-in provider on purpose: existing users' config
# (`addons.options.whispercpp`) and their downloaded models are keyed by it,
# and the host derives the `WHISPERCPP_*` env prefix from it.
ADDON_ID = 'whispercpp'
PROTOCOL_VERSION = 1
