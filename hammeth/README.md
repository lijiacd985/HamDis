# hammeth

A Python toolkit for scWGBS Hamming-distance based analysis.

## Structure
- hammeth/commands: CLI subcommands
- hammeth/scripts: core implementation

## Basic usage
See source code and future docs.

## usage: hammeth [-h] {prepare,hamdist,matrix} ...

positional arguments:
  {prepare,hamdist,matrix}
    prepare             prepare PAT/cell inputs
    hamdist             run Hamming-distance pipeline
    matrix              build ratio beds and Hamming distance matrix

optional arguments:
  -h, --help            show this help message and exit

## GPU acceleration

`hamdist` now supports an optional `--use_gpu` flag to accelerate pairwise
Hamming distance calculation with CUDA (via CuPy). If GPU/CuPy is unavailable
or GPU runtime fails, the pipeline automatically falls back to CPU.
