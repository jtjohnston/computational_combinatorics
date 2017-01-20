#!/usr/bin/env python

import sys
import itertools


def main():
	
	###########################################################################################################
	#########  These few values are constants throughout the computation.  ###################################
	###########################################################################################################
	
	N = int(sys.argv[1])					### The number of elements in the base set.
	K = int(sys.argv[2])					### The number of elements in each subset representing a vertex.
	
	SIZES = map(lambda x: int(x), sys.argv[3:])		### There is an edge between vertex i and j if the intersection
													### of their corresponding sets is a size in SIZES.
													### For Kneser graphs, SIZES=[0].
	print SIZES	
	V = [set(x) for x in itertools.combinations( range(N), K )]	
	NUM_VERTICES = len(V)
	
	
	###########################################################################################################
	###########################################################################################################
	
	
	### Create the adjacency matrix of the graph
	
	A = [ [0 for x in xrange(NUM_VERTICES)] for y in xrange(NUM_VERTICES) ]
	
	for i, j in itertools.product( xrange(NUM_VERTICES), xrange(NUM_VERTICES) ):
		if len( V[i].intersection( V[j] ) ) in SIZES:
			A[i][j] = 1

	with open("matrix_%d_%d.txt" % (N, K), 'w') as outFile:
		for i in xrange(NUM_VERTICES):
			for j in xrange(NUM_VERTICES):
				outFile.write("%d " % A[i][j])
			outFile.write("\n")

main()
