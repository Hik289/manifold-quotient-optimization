#!/usr/bin/env python3
"""Run a command on a user-configured SSH host."""
import os
import shlex
import subprocess
import sys

host = os.environ.get("QUOTIENT_SSH_HOST")
if not host:
    raise SystemExit("Set QUOTIENT_SSH_HOST to your own SSH host")
raise SystemExit(subprocess.call(["ssh", host, shlex.join(sys.argv[1:])]))
