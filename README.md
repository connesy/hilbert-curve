# hilbert-curve
Small project to recursively calculate and plot each successive order of the Hilbert Curve.

After watching [3Blue1Brown's video on Hilbert's Curve](https://www.youtube.com/watch?v=3s7h2MHQtxc) again recently, I decided to code it up myself.
Especially the recursive nature of the definition of each order of the curve seemed like a fun challenge.

# How to run:
Requirements: `pip install -r requirements.txt`

Simply run `python HilbertCurve.py` from a terminal. The script will plot the lowest order Hilbert Curve and wait for any user input in the terminal. On each user input, each successive order is plotted.

You can specify the maximum order that is plotted by running the script with the `--max-order=<order>` flag. Default value for max order is `8`.

### Example plot
![image](https://user-images.githubusercontent.com/13164166/140662805-8ca5ebca-b67e-41b0-b729-9b890685bde6.png)
