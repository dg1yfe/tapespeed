#!/bin/sh
# Pack the stored test data (crates/*/tests/data) into dist/v<N>/tapespeed-testdata.tar.zst,
# reproducibly: sorted file list, mtime 2000-01-01 00:00 UTC, uid/gid 0, modes 644, ustar format,
# no extended attributes, zstd -19 --long. The same tree gives the same bytes (and sha256).
#
#   tools/pack-testdata.sh <N>          build data version N (1, 2, ...), print the lock fields
#   tools/pack-testdata.sh <N> --lock   also rewrite testdata.lock (keeps the mirror URLs, with
#                                       their /v<old>/ directory replaced by /v<N>/)
# Then upload dist/v<N>/tapespeed-testdata.tar.zst to /files/tapespeed/v<N>/ on every mirror and
# commit testdata.lock.
#
# Needs: POSIX sh, tar (GNU tar or bsdtar/libarchive), zstd, sha256sum or shasum.
set -eu
die() { echo "pack-testdata: $*" >&2; exit 1; }
[ $# -ge 1 ] || die "usage: $0 <N> [--lock]"
version=$1
lock_out=${2:-}
case $version in *[!0-9]*|'') die "version must be a number (data version N, URL dir v<N>)" ;; esac
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
for t in tar zstd; do command -v $t >/dev/null || die "$t not found"; done
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
file=tapespeed-testdata.tar.zst
out=dist/v$version
mkdir -p "$out"
stage=$(mktemp -d "${TMPDIR:-/tmp}/ts-pack.XXXXXX")
trap 'rm -rf "$stage"' EXIT INT TERM

# file list: regular files only, byte-order sorted, no hidden files
LC_ALL=C find crates/*/tests/data -type f ! -name '.*' | LC_ALL=C sort > "$stage/list"
[ -s "$stage/list" ] || die "no files under crates/*/tests/data (fetch them first: tools/fetch-testdata.sh)"
# staging copy with normalised metadata (the work tree is not touched)
while IFS= read -r f; do
  mkdir -p "$stage/tree/$(dirname "$f")"
  cp "$f" "$stage/tree/$f"
done < "$stage/list"
find "$stage/tree" -type f -exec chmod 644 {} +
find "$stage/tree" -exec touch -t 200001010000.00 {} +

if tar --version 2>/dev/null | grep -q 'GNU tar'; then
  (cd "$stage/tree" && TZ=UTC0 tar --format=ustar --owner=0 --group=0 --numeric-owner \
     --no-recursion -cf - -T "$stage/list") > "$stage/a.tar"
else
  # bsdtar (macOS, libarchive): -n = no recursion; COPYFILE_DISABLE: no AppleDouble files
  (cd "$stage/tree" && TZ=UTC0 COPYFILE_DISABLE=1 tar --format ustar --uid 0 --gid 0 \
     --uname '' --gname '' --no-xattrs --no-acls --no-fflags -n -cf - -T "$stage/list") > "$stage/a.tar"
fi
zstd -q -19 --long=27 --no-check -T1 -f "$stage/a.tar" -o "$out/$file.part"
mv "$out/$file.part" "$out/$file"
size=$(wc -c < "$out/$file" | tr -d ' ')
sum=$(sha256 "$out/$file")
n=$(wc -l < "$stage/list" | tr -d ' ')
echo "packed $n files -> $out/$file"
echo "version=$version"
echo "file=$file"
echo "size=$size"
echo "sha256=$sum"

if [ "$lock_out" = "--lock" ]; then
  lock=$root/testdata.lock
  {
    echo "# Stored test data (crates/*/tests/data), not in git. tools/fetch-testdata.sh downloads"
    echo "# the archive from the url= lines (tried in order), checks size and sha256 and extracts it."
    echo "# New data version N: tools/pack-testdata.sh N --lock, upload dist/vN/$file to"
    echo "# /files/tapespeed/vN/ on every mirror, commit this file (version, size, sha256, urls together)."
    echo "version=$version"
    echo "file=$file"
    echo "size=$size"
    echo "sha256=$sum"
    if [ -f "$lock" ] && grep -q '^url=' "$lock"; then
      grep '^url=' "$lock" | sed "s#/v[0-9][0-9]*/#/v$version/#"
    else
      echo "url=https://ib-erckenbrecht.de/files/tapespeed/v$version/$file"
    fi
  } > "$lock.tmp"
  mv "$lock.tmp" "$lock"
  echo "wrote $lock"
fi
