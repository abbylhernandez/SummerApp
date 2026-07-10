#!/bin/sh
set -eu

cd "$(dirname "$0")"

# Match the original toolchain: MinGW-w64 gcc, no aggressive FP optimization.
gcc -O0 -shared -o libfunctions.dll function.c -lm

echo "Built libfunctions.dll"
