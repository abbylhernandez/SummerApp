#!/bin/sh
set -eu

cd "$(dirname "$0")"

clang -dynamiclib -O2 -Wall -o libfunctions.dylib function.c -lm

echo "Built libfunctions.dylib"
