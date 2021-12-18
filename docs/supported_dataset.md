## Supported Corpus Type

We support three types of corpus (corpus_type: docu_json, docu_text, sent_text) currently.

## 1. docu_json

It is in the form of a json file composed of document units, and each json object must be composed of one document.

```bash
# sample
{"text":"sent1 sent2 sent3..."}
{"text":"sent4 sent5 sent6..."}
{"text":"sent7 sent8 sent9..."}
...

# example
{"text":"Earth science is a name that combines the studies of the planet Earth. The Earth's environment is largely divided into land, sea, and atmosphere, and each of these environments is the subject of major research in geological, hydrological, and atmospheric science, which can be said to be the main fields of earth science. Studies commonly called geoscience include meteorology targeting phenomena occurring in the atmosphere, geology mainly targeting substances on Earth's surface, oceanography targeting ocean phenomena, and geophysics targeting phenomena occurring deep inside the Earth."}
```

## 2. docu_text

It is in the form of a text file composed of document units, and each document must be divided into one new line ('\n').

```bash
# sample
sent1 sent2 sent3 ...
sent4 sent5 sent6 ...
sent7 sent8 sent9 ...

# example
Earth science is a combination of studies that ...
The Poincare conjecture is a summary of the topological features of a three-dimensional sphere that is the boundary ...
Euclidean geometry is a mathematical system built by the ancient Greek mathematician Euclades (Euclid) ...
```

## 3. sent_text

It is in the form of a txt file composed of sentence units, and each sentence must be divided into one new line ('\n') and each document must be divided into two new lines.

```bash
# sample
sent1 
sent2 
sent3

sent4
sent5
sent6

# example
Transcendental numbers refer to numbers that are not algebraic in mathematics.
It means a non-zero root with a complex and rational coefficient that no polynomial equation can do.
The most well-known transcendental numbers are (the rate of origin) and (below the natural logarithm).

The Maxwell equation shows that light is also one of the electromagnetic waves.
Each equation is called the Gaussian Law, the Gaussian Law of Magnetism, the Faraday Law of Electromagnetic Induction, and the Angper Circuit Law.
After James Clark Maxwell put each equation together, it was called the Maxwell equation.
```
