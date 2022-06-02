#!/bin/bash


python3 run_sims.py -anneal -platform opencl -name test -dt 0.001 -ftype input_files/type_table.csv -ftop input_files/DLD_chr10_top.txt  -fseq input_files/DLD_seq_chr10.txt -rep 1 -Ta 1 -G 2676 -F 0 -temp 120 -kb 5 -Esoft 5 -nblocks 50 -blocksize 100 -R0 30 -outpath test_out/
