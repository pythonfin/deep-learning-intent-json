# deep-learning-intent-json
This is a NLP deep learning Python program using Keras / Tensorflow 2.

It uses a bi-directional transformer model for natural language processing.

Coded in a professional style to meet PEP8 standards. You can gain a 
greater understanding of deep learning transformer NLP structure by reviewing
the code and comments.

Currently structured using a JSON input file to match sentences with the 
appropriate intent. Hence the intent determination logic coded within. 
More file upload options will be added later.

Basic JSON format and a few examples are given. Actual JSON file is much 
larger for production use.

Uses the following Python modules:
re - regular expressions
Numpy
Tensorflow 2
Keras
matplotlib
JSON
poetry

easy install w Pycharm and Poetry
Open your project
create a poetry environment in PyCharm

then in your terminal console in Pycharm:
$ python -m poetry lock --no-update
$ python -m poetry install

key packages:
$ poetry add numpy
$ poetry add matplotlib
$ poetry add tensorflow
$ poetry add tensorflow_datasets

check tensorflow was added correctly by:
$ python -m pip show tensorflow

