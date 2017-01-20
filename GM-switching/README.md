# Godsil-McKay Switching in Johnson Scheme Graphs

The code is this repo was used, in part, in the experimentation that led to the paper:

Title: Cospectral mates the for union of some classes in the Johnson association scheme

Authors: Sebastian M. Ciaba, Matt McGinnis, and Travis Johnston


## Using the code

The directions provided assume:
- you are using a linux command-line;
- you have python 2.7 installed;
- you have a CUDA-enabled GPU with compute capability 3.0+;
- and, you have installed CUDA (and have access to the nvcc compiler).


### generate_johnson_graph.py

generate_johnson_graph.py creates an adjacency matrix of a Johnson graph.
These graphs have as vertices the k-subsets of an n-element set.
The user of the code specifies n and k.
Two vertices are adjacent it the size of their intersection is in a set S.
The user specifies the (positive) integers in the set S.
For example, to create the Kneser(8, 3) graph (S={0}) the use could type:

python generate_johnson_graph.py 8 3 0

or, if the code is made executable with chmod +x,

./generate_johnson_graph.py 8 3 0

The code echoes (as a list) the set S, in this case [0].
It then creates a file, in the current directory called: matrix_8_3.txt.
This file contains the adjacency matrix (in plain-text readable format) of the Johnson graph.
This code is intended to produce the input files for switching.cu which we discuss next.
Note that if S is more than just a single integer, the user just provides a space separated
list of integers.  The general format is:

python generate_johnson_graph.py n k s1 s2 ... sm

Where n is the size of the base set, k is the size of the subsets, and S={s1, s2, ..., sm}.


