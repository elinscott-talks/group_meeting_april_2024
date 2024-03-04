#!/bin/bash
rsync -rav --include=*/ --include='*.png' --exclude='*' --prune-empty-dirs  theossrv8:/scratch/elinscott/projector_tests/ .
