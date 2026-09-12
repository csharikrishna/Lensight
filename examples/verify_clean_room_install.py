"""
Test clean-room installation of lensight wheel in an isolated temporary virtual environment.
Verifies:
1. Wheel installation from dist/
2. CLI binary registration ('lensight' executable in Scripts/bin)
3. CLI commands: lensight --help, lensight info, lensight audit --help, lensight leakage --help, lensight clean --help, lensight help
"""
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

def test_clean_install():
    repo_root = Path(__file__).resolve().parent.parent
    wheel_files = list((repo_root / "dist").glob("lensight-0.2.0-py3-none-any.whl"))
    assert wheel_files, "Wheel file not found in dist/"
    wheel_path = wheel_files[0]
    print(f"Testing wheel: {wheel_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_dir = Path(tmpdir) / "test_venv"
        print(f"Creating venv at: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)

        # Determine python and lensight executables
        if sys.platform == "win32":
            python_bin = venv_dir / "Scripts" / "python.exe"
            lensight_bin = venv_dir / "Scripts" / "lensight.exe"
        else:
            python_bin = venv_dir / "bin" / "python"
            lensight_bin = venv_dir / "bin" / "lensight"

        # Install wheel with dependencies
        print("Installing wheel...")
        res = subprocess.run(
            [str(python_bin), "-m", "pip", "install", str(wheel_path)],
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            print("STDERR:", res.stderr)
            print("STDOUT:", res.stdout)
            raise RuntimeError("Pip install failed")

        print(f"Checking CLI binary exists: {lensight_bin}")
        assert lensight_bin.exists(), f"CLI executable not created at {lensight_bin}"

        # Test CLI commands
        test_commands = [
            [str(lensight_bin), "--help"],
            [str(lensight_bin), "info"],
            [str(lensight_bin), "audit", "--help"],
            [str(lensight_bin), "leakage", "--help"],
            [str(lensight_bin), "clean", "--help"],
            [str(lensight_bin), "help"],
        ]

        for cmd in test_commands:
            print(f"Running: {' '.join(cmd)}")
            out = subprocess.run(cmd, capture_output=True, text=True)
            assert out.returncode == 0, f"Command failed: {' '.join(cmd)}\nStderr: {out.stderr}\nStdout: {out.stdout}"
            assert "lensight" in out.stdout.lower() or "audit" in out.stdout.lower() or "lensight" in (venv_dir.name).lower()
            print(f"  -> SUCCESS (exit code 0)")

    print("\n Clean-room install verified successfully! All entry points work.")

if __name__ == "__main__":
    test_clean_install()
