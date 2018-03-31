# Mandelbrot calculator
Produce beautiful fraktals with Python
Fast and little dependencies

## Dependencies

* Python3
* numba
* numpy, matplotlib, argparse


## Examples

```bash
$ time python3 mandelbrot.py -o example_bg4k.png --maxiter 1000 --saturation=8 point --point=-.736:-.2086 -R 3840x2160 -F 10
python3 mandelbrot.py -o example_bg4k.png --maxiter 1000 --saturation=8 point    12,38s user 0,47s system 99% cpu 12,959 total
```
[<img src="examples/example_bg4k_small.png">](examples/example_bg4k.png)
