#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

tag=b12499
commit=401563f10ec60754e688cd3165c1343cf4be9642
digest=c4cb6ddc6cb499de4beba4512922338627daf4d8d5fa6a488a2502e0839db8e1
mkdir -p .tools
if [[ ! -d .tools/llama.cpp/.git ]]; then
    git clone --quiet --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp.git .tools/llama.cpp
fi
[[ "$(git -C .tools/llama.cpp rev-parse HEAD)" == "$commit" ]] || {
    echo "The converter must be at llama.cpp $commit." >&2
    exit 1
}
if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
    echo "Build llama.cpp $tag for your platform and place its executables in .tools/llama-bin." >&2
    exit 1
fi
archive=".tools/llama-$tag.tar.gz"
if ! echo "$digest  $archive" | shasum -a 256 -c --status 2>/dev/null; then
    curl -fsSL --retry 3 "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-macos-arm64.tar.gz" -o "$archive.part"
    echo "$digest  $archive.part" | shasum -a 256 -c -
    mv "$archive.part" "$archive"
fi
mkdir -p .tools/llama-bin
tar -xzf "$archive" --strip-components=1 -C .tools/llama-bin
