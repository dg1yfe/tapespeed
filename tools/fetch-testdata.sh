#!/bin/sh
# Fetch the stored test data (crates/*/tests/data) named in testdata.lock.
#
#   tools/fetch-testdata.sh
#
# - Data present and matching the lock (.testdata.stamp = the lock's sha256): nothing to do.
# - Otherwise the archive comes from, in this order:
#     1. $TAPESPEED_TESTDATA (a local .tar.zst; an offline override, no download);
#     2. dist/v<N>/<file> (left by tools/pack-testdata.sh or an earlier download; CI caches it);
#     3. the url= lines of testdata.lock, tried in order (curl).
#   Its size and sha256 are checked against the lock, then the crates/*/tests/data directories in
#   the archive replace the ones in the work tree and the stamp is written.
#
# Needs only POSIX sh, curl, zstd, tar and sha256sum (Linux, coreutils) or shasum (macOS).
# macOS: `brew install zstd` (curl, tar, shasum ship with the system). Debian/Ubuntu:
# `apt install curl zstd` (tar, sha256sum are in the base system).
set -eu
die() { echo "fetch-testdata: error: $*" >&2; exit 1; }
say() { echo "fetch-testdata: $*"; }
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
lock=testdata.lock
[ -f "$lock" ] || die "$root/$lock not found"
field() { sed -n "s/^$1=//p" "$lock" | head -n 1; }
version=$(field version)
file=$(field file)
size=$(field size)
sha=$(field sha256)
[ -n "$version" ] && [ -n "$file" ] && [ -n "$size" ] && [ -n "$sha" ] ||
  die "$lock: needs version=, file=, size=, sha256= lines"
stamp=.testdata.stamp

if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sha" ]; then
  say "test data v$version present (sha256 ${sha%"${sha#????????????}"}...), nothing to do"
  exit 0
fi

for t in zstd tar; do
  command -v "$t" >/dev/null || die "'$t' not found (macOS: brew install zstd; Debian/Ubuntu: apt install zstd)"
done
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  else die "neither sha256sum nor shasum found"; fi
}
# check <path>: size and sha256 against the lock; prints why not on stderr
check() {
  n=$(wc -c < "$1" | tr -d ' ')
  if [ "$n" != "$size" ]; then
    echo "fetch-testdata: $1: size $n, testdata.lock says $size" >&2
    return 1
  fi
  s=$(sha256 "$1")
  if [ "$s" != "$sha" ]; then
    echo "fetch-testdata: $1: sha256 $s, testdata.lock says $sha" >&2
    return 1
  fi
}

cache=dist/v$version/$file
archive=
if [ -n "${TAPESPEED_TESTDATA:-}" ]; then
  [ -f "$TAPESPEED_TESTDATA" ] || die "TAPESPEED_TESTDATA=$TAPESPEED_TESTDATA: no such file"
  check "$TAPESPEED_TESTDATA" || die "TAPESPEED_TESTDATA=$TAPESPEED_TESTDATA does not match $lock (corrupt or another version)"
  archive=$TAPESPEED_TESTDATA
  say "using $archive"
elif [ -f "$cache" ] && check "$cache"; then
  archive=$cache
  say "using $archive"
else
  command -v curl >/dev/null || die "'curl' not found"
  urls=$(sed -n 's/^url=//p' "$lock")
  [ -n "$urls" ] || die "$lock has no url= lines; set TAPESPEED_TESTDATA=/path/$file"
  mkdir -p "dist/v$version"
  tried=
  for u in $urls; do
    say "downloading $u"
    if curl -fL --retry 2 --connect-timeout 20 -sS -o "$cache.part" "$u"; then
      if check "$cache.part"; then
        mv "$cache.part" "$cache"
        archive=$cache
        break
      fi
    fi
    rm -f "$cache.part"
    tried="$tried
  $u"
  done
  [ -n "$archive" ] || die "no mirror delivered a valid $file (v$version, sha256 $sha). Tried:$tried
Download it by hand and run: TAPESPEED_TESTDATA=/path/to/$file tools/fetch-testdata.sh"
fi

# extract into a scratch directory first, then swap the data directories in
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ts-fetch.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM
zstd -dc --long=27 "$archive" | tar -xf - -C "$tmp" || die "cannot extract $archive"
[ -d "$tmp/crates" ] || die "$archive: no crates/ directory inside"
rm -f "$stamp"
for d in "$tmp"/crates/*/tests/data; do
  [ -d "$d" ] || die "$archive: no crates/*/tests/data inside"
  rel=${d#"$tmp"/}
  [ -d "$(dirname "$rel")" ] || die "$rel: crate $(echo "$rel" | cut -d/ -f2) not in this work tree"
  rm -rf "$rel"
  mv "$d" "$rel"
done
printf '%s\n' "$sha" > "$stamp"
n=$(find crates/*/tests/data -type f | wc -l | tr -d ' ')
say "test data v$version installed ($n files in crates/*/tests/data)"
