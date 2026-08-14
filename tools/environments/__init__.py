"""Hermes execution environment backends.

Each backend provides the same interface (BaseEnvironment ABC) for running
shell commands in a specific execution context: local, Docker, Singularity,
SSH, Modal, Daytona, or Harbor (trial-scoped; do not set in config.yaml).

The terminal_tool.py factory (_create_environment) selects the backend
based on the TERMINAL_ENV configuration.
"""

from tools.environments.base import BaseEnvironment

__all__ = ["BaseEnvironment"]
