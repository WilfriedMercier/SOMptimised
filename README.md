# SOMptimised
An optimised version of [sklearn-som](https://pypi.org/project/sklearn-som/) with extended features.

Additional features:

* Can save additional features into the SOM beyond those used to train it
* Can serialise (i.e. save SOM state into a binary file onto the disk)
* Can load back the SOM in its previous state from a binary file on the disk

This SOM implementation has been optimised in terms of speed with respect to [sklearn-som](https://pypi.org/project/sklearn-som/) just by using more efficient numpy functions and features and by reducing the number of loops when possible.

Performance boost does not scale linearly with SOM or dataset size but, as an indication, a 50x50 SOM run on 14 000 data points (1 epoch) takes on my machine:

* **7.3s of CPU and wall time to fit with this library**
* 2min of CPU and wall time to fit with [sklearn-som](https://pypi.org/project/sklearn-som/)

For more details, please visite the [documentation](https://wilfriedmercier.github.io/SOMptimised/index.html).

# License

MIT License

Copyright (c) 2023 Wilfried Mercier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
