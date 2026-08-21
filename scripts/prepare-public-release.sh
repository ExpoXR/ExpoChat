#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/new-public-repository" >&2
  exit 2
fi

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_root="$1"

if [[ "$release_root" != /* || "$release_root" == "/" || -e "$release_root" ]]; then
  echo "release destination must be a new absolute path" >&2
  exit 2
fi

python3 "$source_root/scripts/public_release_audit.py"

staging_root="$(mktemp -d)"
trap 'rm -rf -- "$staging_root"' EXIT

cd "$source_root"
while IFS= read -r -d '' path; do
  [[ -f "$path" || -L "$path" ]] && printf '%s\0' "$path"
done < <(git ls-files --cached --others --exclude-standard -z) \
  | tar --null --files-from=- --create --file=- \
  | tar --extract --file=- --directory="$staging_root"

git -C "$staging_root" init --initial-branch=main
git -C "$staging_root" add --all
git -C "$staging_root" \
  -c user.name="ExpoXR Release" \
  -c user.email="hallo@expoxr.com" \
  commit --message="Initial public release"
python3 "$staging_root/scripts/public_release_audit.py" --history

mv -- "$staging_root" "$release_root"
trap - EXIT
echo "Clean public repository prepared at $release_root"
