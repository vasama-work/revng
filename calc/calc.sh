#!/bin/bash

bin/revng clift-opt /dev/null \
  --import-llvm="llvm=/home/vasama/c/revng/orchestra/sources/revng/calc/calc.ll model=/home/vasama/c/revng/orchestra/sources/revng/calc/m.yml" \
  --beautify \
  --emit-c="tagless model=/home/vasama/c/revng/orchestra/sources/revng/calc/m.yml output=/home/vasama/c/revng/orchestra/sources/revng/calc/calc.c" \
  -o /home/vasama/c/revng/orchestra/sources/revng/calc/calc.mlir