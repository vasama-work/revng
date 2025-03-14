#!/usr/bin/bash

SAMPLE_PATH="/home/vasama/c/revng/orchestra/sources/revng/examples"
SAMPLE_IR_PATH="$SAMPLE_PATH/$1.ll"
SAMPLE_MODEL_PATH="$SAMPLE_PATH/model.yml"

bin/revng clift-opt \
	--import-llvm="llvm=$SAMPLE_IR_PATH model=$SAMPLE_MODEL_PATH" \
	--emit-c="tagless model=$SAMPLE_MODEL_PATH" \
	/dev/null -o /dev/null
