import os
import re

from setuptools import setup

if __name__ == "__main__":
    try:
        PKG = "cvar_sensing"

        # Configure version.
        def find_version() -> str:
            def read(fname: str) -> str:
                return open(os.path.realpath(os.path.join(os.path.dirname(__file__), fname)), encoding="utf8").read()

            version_file = read(f"src/{PKG}/version.py").split("\n")[0]
            version_re = r"__version__ = \"(?P<version>.+)\""
            version_raw = re.match(version_re, version_file)

            if version_raw is None:
                raise RuntimeError(f"__version__ value not found, check src/{PKG}/version.py")

            version = version_raw.group("version")
            return version

        setup(
            version=find_version(),
        )
    except:  # noqa
        print("\n\nAn error occurred while building the project")  # noqa: T201
        raise
