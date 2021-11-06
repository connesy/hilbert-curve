# hilbert-curve
Small project to recursively calculate and plot each successive order of the Hilbert Curve.

After watching [3Blue1Brown's video on Hilbert's Curve](https://www.youtube.com/watch?v=3s7h2MHQtxc) again recently, I decided to code it up myself.
Especially the recursive nature of the definition of each order of the curve seemed like a fun challenge.

# How to run:
Simply run `HilbertCurve.py` from a terminal. The script will plot the lowest order Hilbert Curve and wait for any user input in the terminal. On each user input, each successive order is plotted.

You can specify the maximum order that is plotted by running the script with the `--max-order=<order>` flag. Default value for max order is `8`.
