"""Modular implementations used by the thin argparse compatibility facade."""

from qqa.commands.runtime import command_version, print_score, resolve_device
from qqa.commands.system import command_doctor, command_gui, resolve_streamlit_app

__all__ = [
    "command_doctor",
    "command_gui",
    "command_version",
    "print_score",
    "resolve_device",
    "resolve_streamlit_app",
]
