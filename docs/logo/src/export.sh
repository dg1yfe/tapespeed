#!/bin/zsh
# Regenerate the tapespeed logo family: SVGs, PNG/ICO exports, contact sheet.
# usage:  docs/logo/src/export.sh [PYTHON]
#   PYTHON     interpreter with shapely >= 2 installed (default: python3)
# needs: rsvg-convert (librsvg), ImageMagick 7 (magick)
set -e
export PYTHONDONTWRITEBYTECODE=1
HERE=${0:A:h}; L=${HERE:h}; PY=${1:-python3}
$PY $HERE/build.py $L
P=$L/png; mkdir -p $P
T=$(mktemp -d)
Z=(-strip -define png:compression-level=9)
# C icon, transparent, for dark and light UIs
for n in 16 32 48 64 128 256 512; do
  rsvg-convert -w $n -h $n $L/C/icon.svg       | magick - $Z $P/icon-dark-$n.png
  rsvg-convert -w $n -h $n $L/C/icon-light.svg | magick - $Z $P/icon-light-$n.png
done
# favicon + apple-touch: C icon on the brand black, opaque, small margin
tile() { # size inner out
  rsvg-convert -w $2 -h $2 $L/C/icon.svg | magick - -background '#0b0b0b' -gravity center -extent ${1}x${1} $Z $3
}
tile 16 16 $T/f16.png; tile 32 30 $T/f32.png; tile 48 44 $T/f48.png
magick $T/f16.png $T/f32.png $T/f48.png $P/favicon.ico
tile 180 150 $P/apple-touch-icon.png
# README header (transparent) and social preview
rsvg-convert -w 1600 $L/A/emblem.svg       | magick - $Z $P/header-dark.png
rsvg-convert -w 1600 $L/A/emblem-light.svg | magick - $Z $P/header-light.png
rsvg-convert -w 1140 $L/A/emblem.svg | magick - -background '#0b0b0b' -gravity center -extent 1280x640 $Z $P/social-preview.png
# contact sheet
cp -R $L/A $L/B $L/C $T/
$PY $HERE/sheet.py $T > /dev/null
rsvg-convert $T/sheet.svg | magick - +dither -colors 256 $Z $L/contact-sheet.png
rm -rf $T
echo "done: $L"
