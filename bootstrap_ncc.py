"""Install the exact Windows nncase release used for KibbleLM."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

ASSET = "https://github.com/kendryte/nncase/releases/download/v0.2.0-beta4/ncc_win_x86_64.zip"
SHA256 = "a38d044052f33d71eee3ce70ef1022eb0aea2b4ffef62ad251e0dea74cf4f5cd"


def main():
    tools = Path(".tools")
    tools.mkdir(exist_ok=True)
    archive = tools / "ncc_win_x86_64.zip"
    temporary = archive.with_suffix(".part")
    source = archive
    if not source.exists():
        urllib.request.urlretrieve(ASSET, temporary)
        source = temporary
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"nncase archive SHA-256 mismatch: {source}")
    if source == temporary:
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as release:
        release.extractall(tools / "ncc")


if __name__ == "__main__":
    main()
