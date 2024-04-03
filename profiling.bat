py -3.10 -B -m cProfile -o .\output\profile.prof .\simple_swcam.py
py -3.10 -m snakeviz .\output\profile.prof