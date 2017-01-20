# Godsil-McKay Switching in Johnson Scheme Graphs

The code is this repo was used, in part, in the experimentation that led to the paper:

**Title:** *Cospectral mates the for union of some classes in the Johnson association scheme*

**Authors:** Sebastian M. Cioaba, Matt McGinnis, and Travis Johnston


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

`python generate_johnson_graph.py 8 3 0`

or, if the code is made executable with chmod +x,

`./generate_johnson_graph.py 8 3 0`

The code echoes (as a list) the set S, in this case [0].
It then creates a file, in the current directory called: matrix_8_3.txt.
This file contains the adjacency matrix (in plain-text readable format) of the Johnson graph.
This code is intended to produce the input files for switching.cu which we discuss next.
Note that if S is more than just a single integer, the user just provides a space separated
list of integers.  The general format is:

`python generate_johnson_graph.py n k s1 s2 ... sm`

Where n is the size of the base set, k is the size of the subsets, and S={s1, s2, ..., sm}.


### switching.cu

switching.cu uses nVidia's CUDA C/C++ and must be compiled prior to use.
Because of the number of threads the main kernel generates the code may fail if it is not 
compiled for and used on a CUDA enabled GPU with compute capability 3.0 or greater.
If the graph (and subsets) are small it can be safely used on an older GPU.
To compile the code type:

`nvcc switching.cu --gpu-architecture=sm_30`

This will compile the code and create an executable called a.out.
The compiled code requires 5 parameters:
- N, the number of vertices in the graph (n choose k if it is a Johnson graph)
- K, the size of the subsets to examine for switching sets
- n, the same n used in the creation of the Johnson graph
- k, the same k used in the creation of the Johnson graph
- g, the GPU number to use.

Kneser(8, 3) has 56 vertices.
We can explore these vertices for potential switching sets of size 6 by running:

`./a.out 56 6 8 3 0`

(which runs the code on GPU 0).
Running the code produces the output:

```
Checking 3478761 subsets.
CUDA error: no error
Examining subsets 0 to 999999999.
CUDA error: no error
Possible switching set 0: { 0 1 2 } { 0 1 3 } { 0 1 4 } { 0 1 5 } { 0 1 6 } { 0 1 7 } 
Possible switching set 1360001: { 0 1 2 } { 0 2 3 } { 0 2 4 } { 0 2 5 } { 0 2 6 } { 0 2 7 } 
Possible switching set 3154129: { 0 1 2 } { 1 2 3 } { 1 2 4 } { 1 2 5 } { 1 2 6 } { 1 2 7 } 
Done.
```

`Examining subsets 0 to 999999999` appears to update the user on progress when more than 1B subsets
are checked.  In the case of the code presented, after checking 3478761 subsets no further checks are made.
Note that the code (as written) assumes the graph is vertex transitive, so it only explores subsets that
include vertex 0 ({0, 1, 2, ..., k-1}).
