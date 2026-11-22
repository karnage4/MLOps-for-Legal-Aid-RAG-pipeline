"""The Azure deploy script's key reader repairs common paste mistakes.

The first Azure deployment crash-looped because a key was stored with the
quotes from a local key file. ``deploy/azure/read-key.sh`` now cleans and
checks each pasted key before it reaches Azure.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "azure" / "read-key.sh"


def _find_bash() -> str | None:
    if sys.platform != "win32":
        return shutil.which("bash")
    # On Windows, bash on PATH may be the WSL launcher, which cannot read
    # Windows paths; use the bash that ships with Git for Windows instead.
    git = shutil.which("git")
    if git:
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.exists():
            return str(candidate)
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")


def run_read_key(pasted, name="PINECONE_API_KEY", prefix="pcsk_", **extra_env):
    env = {key: value for key, value in os.environ.items() if key != "CHECK_KEY_PREFIXES"}
    env.update(extra_env)
    return subprocess.run(
        [BASH, "-c", f'source "$1"; read_key {name} {prefix}', "bash", HELPER.as_posix()],
        input=pasted.encode(),
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    "pasted",
    [
        "pcsk_test\n",
        '"pcsk_test"\n',
        "  'pcsk_test'  \n",
        'PINECONE_API_KEY="pcsk_test"\r\n',
        "export PINECONE_API_KEY=pcsk_test\n",
        "PINECONE_API_KEY = pcsk_test\n",
    ],
)
def test_paste_mistakes_are_cleaned(pasted):
    result = run_read_key(pasted)

    assert result.returncode == 0
    assert result.stdout.decode() == "pcsk_test"
    stderr = result.stderr.decode()
    assert "accepted: 9 characters, starts with 'pcsk_'" in stderr
    assert "pcsk_test" not in stderr


def test_wrong_key_is_asked_again():
    result = run_read_key("gsk_test\npcsk_test\n")

    assert result.returncode == 0
    assert result.stdout.decode() == "pcsk_test"
    stderr = result.stderr.decode()
    assert "keys start with 'pcsk_'" in stderr
    assert "gsk_test" not in stderr


@pytest.mark.parametrize("pasted", ["\n\n\n", "x\ny\nz\n", "pcsk_a pcsk_b\n" * 3])
def test_three_rejected_pastes_fail(pasted):
    result = run_read_key(pasted)

    assert result.returncode == 1
    assert result.stdout.decode() == ""
    assert "was not accepted after 3 attempts" in result.stderr.decode()


def test_groq_key_uses_its_own_name_and_prefix():
    result = run_read_key('GROQ_API_KEY="gsk_test"\n', name="GROQ_API_KEY", prefix="gsk_")

    assert result.returncode == 0
    assert result.stdout.decode() == "gsk_test"


def test_prefix_check_can_be_skipped_for_older_key_formats():
    result = run_read_key("legacy-key-format\n", CHECK_KEY_PREFIXES="0")

    assert result.returncode == 0
    assert result.stdout.decode() == "legacy-key-format"
    stderr = result.stderr.decode()
    assert "prefix check skipped" in stderr
    assert "legacy" not in stderr
