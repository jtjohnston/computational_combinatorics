#include<stdio.h>
#include<stdlib.h>
#include<fstream>
#include<algorithm>

// This value is the largest subsets we must unrank.
// This value must be set before compiling because the size of arrays
// must be fixed (not dynamic) on the GPU.
#define LARGEST_SUBSET 10

using namespace::std;

// Device code to compute the binomial coefficient "n choose k."
// This code is patterned off of: http://stackoverflow.com/a/3025547/4187033
// This function is callable from the device (GPU) and runs on the device (GPU).
__device__ long binom_d(int n, int k){
	long ntok = 1;
	long ktok = 1;
	if(k > n){
		return 0;
	}
	int i, j;
	j = n;
	for(i=1; i<=min(k, n-k); i++){
		ntok *= j;	//A falling factorial (top), n (n-1) (n-2) ... 
		ktok *= i;	//A rising factorial (bottom), 1 (2) (3) ...
		j--;
	}
	return ntok / ktok;
}

// The code is identical to binom_d except that it runs on the CPU (instead of GPU).
long binom_h(int n, int k){
	long ntok = 1;
	long ktok = 1;
	if(k > n){
		return 0;
	}
	int i, j;
	j = n;
	for(i=1; i<=min(k, n-k); i++){
		ntok *= j;
		ktok *= i;
		j--;
	}
	return ntok / ktok;
}



// This code generates a k-subset of an n-element set [0, 1, ..., n-1] and stores it in Kset.
// It finds the "initial_value" k-subset in lexicographic order.
// This code is called from the device (GPU) and runs on the device (GPU).
__device__ void unrank_combination(int n, int k, long initial_value, int* Kset) {
	long cash_on_hand = initial_value;
	int digit;
	long cost_to_increment;
	Kset[0] = 0;	//Initialize the first element.
					//Each of the following elements will start off one bigger than the previous element.
	//Use the cash_on_hand value to "pay" for incrementing each digit.
	//Pay 1-unit for each combination that is "skipped" over.
	//E.g. To increment the 0 in 0, 1, 2, ..., k-1 to a 1 (and force the others to increment to 2, 3, ..., k)
	//it would cost binom(n-1, k-1) since we skipped over each combination of the form
	// 0 * * * ... * and there are binom(n-1, k-1) of those combinations
	for(digit=0; digit<k-1; digit++){
		//There are n-1-Kset[digit] elements left to choose from.
		//Those elements must be used to fill k-1-digit places.
		cost_to_increment = binom_d( n-1-Kset[digit], k-1-digit );
		while(cost_to_increment <= cash_on_hand){
			Kset[digit]++;
			cash_on_hand = cash_on_hand - cost_to_increment;
			cost_to_increment = binom_d( n-1-Kset[digit], k-1-digit );
		}
		Kset[digit+1] = Kset[digit]+1;	//Ititialize the next element of Kset making sure the elements
										//come in sorted order.
	}
	//Kset[k-1] has been initialized to Kset[k-2]+1 (last step).
	//Now, if there is anything left to pay, we simply increment Kset[k-1] by this amount.
	Kset[k-1] += cash_on_hand;
}

// This code is identical to unrank_combination except that it is called by the CPU and runs on CPU.
// The code was originally intended for debugging, but is also helpful for displaying individual subsets of interest.
void unrank_combination_h(int n, int k, long initial_value, int* Kset) {
	long cash_on_hand = initial_value;
	int digit;
	long cost_to_increment;
	Kset[0] = 0;	//Initialize the first element.
					//Each of the following elements will start off one bigger than the previous element.
	//Use the cash_on_hand value to "pay" for incrementing each digit.
	//Pay 1-unit for each combination that is "skipped" over.
	//E.g. To increment the 0 in 0, 1, 2, ..., k-1 to a 1 (and force the others to increment to 2, 3, ..., k)
	//it would cost binom(n-1, k-1) since we skipped over each combination of the form
	// 0 * * * ... * and there are binom(n-1, k-1) of those combinations
	for(digit=0; digit<k-1; digit++){
		//There are n-1-Kset[digit] elements left to choose from.
		//Those elements must be used to fill k-1-digit places.
		cost_to_increment = binom_h( n-1-Kset[digit], k-1-digit );
		while(cost_to_increment <= cash_on_hand){
			Kset[digit]++;
			cash_on_hand = cash_on_hand - cost_to_increment;
			cost_to_increment = binom_h( n-1-Kset[digit], k-1-digit );
		}
		Kset[digit+1] = Kset[digit]+1;	//Ititialize the next element of Kset making sure the elements
										//come in sorted order.
	}
	//Kset[k-1] has been initialized to Kset[k-2]+1 (last step).
	//Now, if there is anything left to pay, we simply increment Kset[k-1] by this amount.
	Kset[k-1] += cash_on_hand;
}



// This function is the heart of the search.
// When this code is called, many threads are spawned on the GPU.
// Each thread is assigned a number (my_subset).
// Each thread unranks the subset assigned by my_subset, and determines whether or not it is a potential switching set.
// If the induced subgraph (on the specific subset of vertices) is regular AND
// If every vertex outside the subraph is adjacent to either 0, 1/2 or all vertices in the subgraph AND
// If there exists a vertex adjacent to exactly 1/2 of all vertices in the subgraph,
// Then, the subset of vertices is reported as a potential switching set.

__global__ void examine_subsets(int n, int k, long MAX, short* A, short* Results, long offset){
	const long my_subset = threadIdx.x + blockIdx.x*blockDim.x + offset;
	const int my_index = threadIdx.x + blockIdx.x*blockDim.x;
	if( my_subset < MAX ){		//MAX = number of subsets to examine.  
								//We make this check because more threads may be spawned then there are subsets that need to be checked.
		int i, j;
		int Kset[LARGEST_SUBSET];	//Kset will store the indices of the vertices this thread will look at.
		for(i=0; i<k; i++){
			Kset[i] = 0;
		}

		//urank_combination modifies Kset to be the specified vertices that should be examined by this thread.
		unrank_combination(n, k, my_subset, Kset);
		
		//induced_subgraph is the adjacency matrix of the subgraph induced by the vertices in Kset.
		//All 2D matrices are stored in 1D form--i.e. entry ij of an (m x n) matrix is in position i*N+j.
		short induced_subgraph[LARGEST_SUBSET*LARGEST_SUBSET];
		for(i=0; i<k; i++){
			for(j=0; j<k; j++){
				induced_subgraph[i*k + j] = A[ Kset[i]*n + Kset[j] ];	//A is n by n (in 1D form) but local_A is k by k (in 1D form)
			}
		}

		//Now, we need to check if induced_subgraph is a potential switching set.
		//First, we need to check if it is regular...
		//To do this, we check if every row sum is the same.
		Results[my_index] = 1;		//Initialize to believing this is a switching set.
									//We change this to zero if we find any reason to know that it isn't one.
		short first_row_sum = 0;
		short this_row_sum = 0;
		for(j=0; j<k; j++){
			first_row_sum += induced_subgraph[j];
		}
		for(i=1; i<k; i++){
			this_row_sum = 0;
			for(j=0; j<k; j++){
				this_row_sum += induced_subgraph[i*k + j];
			}
			if(this_row_sum != first_row_sum){
				Results[my_index] = -1;			//Not regular, hence, not a switching set.
			}
		}	//End check for subgraph regularity.
		

		//If the graph is regular, check to see if it is a switching candidate.
		//It is NOT switching if any vertex (outside of Kset) is not adjacent to either 0, 1/2, or all of these vertices.
		//It is also NOT switching if there isn't at least one vertex (outside of Kset) adjacent to 1/2 of these vertices.
		if(Results[my_index]==1){
			Results[my_index] = 2;		//This flag notes that we have not yet found a vertex adjacent to 1/2.
										//If this is not 0 and we find a vertex adjacent to 1/2, then it becomes a 1.
										//If this ever becomes 0, it stays 0.
			for(i=0; i<n; i++){
				//How many vertices in Kset is vertex i adjacent to?
				//To save registers, recycle first_row_sum for the number of vertices seen and this_row_sum as an indicator for whether or not i is in Kset.
				first_row_sum = 0;
				this_row_sum = 0;
				for(j=0; j<k; j++){
					first_row_sum += A[ i*n + Kset[j] ];		// i'th row, column Kset[j]... 
																// 1 if vertex i is adjacent to Kset[j] (the j'th element of our potential switching set).
					if(i == Kset[j]){
						this_row_sum = 1;						// This indicates that vertex i is in the Kset (and should not be considered).
					}
				}
				if(this_row_sum == 0){	//this_row_sum indicates whether or not vertex i is in Kset.
					//i is not in Kset (since the indicator is 0).
					//Check to see if i is adjacent to 0, 1/2 or all of Kset.
					if( (first_row_sum != 0) && (first_row_sum != k) && (first_row_sum != (k/2)) ){
						Results[my_index]=0;
					}else{
						if( (first_row_sum == (k/2)) && Results[my_index]==2){
							Results[my_index]=1;
						}
					}
				}
			}
		}//End if(Results[my_index]==1) -- the check for outside vertices adjacent to 0, 1/2 or all of Kset
	}//End of if(my_subset < MAX)
}






int main(int argc, char** argv){
	int N = atoi( argv[1] );	//Number of vertices in the Johnson graph;
								//Should be the number of lines in matrix_NN_KK.txt (data file).
	int K = atoi( argv[2] );	//Size of the subset to inspect.
	
	int NN = atoi( argv[3] );	
	int KK = atoi( argv[4] );	//The vertices of the graph are KK-subsets of an NN-element set.
	int GPU = atoi( argv[5] );	//Which GPU to use (can set this to be 0 if you only have a single GPU).


	int* Kset;
	Kset = (int *) malloc( K*sizeof(int) );

	int* sKset;
	sKset = (int *) malloc( KK*sizeof(int) );

	char fileName[50];
	//Expect an adjaceny matrix in current working directory with name matrix_NN_KK.txt (where NN and KK are appropriately replaced).
	sprintf(fileName, "matrix_%d_%d.txt", NN, KK);


	short* h_A;		//A pointer to host memory where the adjacency matrix will be stored
	short* d_A;		//A pointer to device (GPU) memory where the adjaceny matrix will be stored on the GPU
	short* h_Results;	//A list of indicators (potential switching or not) stored on host
	short* d_Results;	//A list of indicators (potential switching or not) stored on the device (GPU).

	const long BATCH = 1000000000;	//The number of threads to spawn at one time; it is also the amount of memory (approx) that will be allocated on the GPU.

	long number_of_subsets = binom_h(N-1, K-1);		//Because of vertex transitivity, only check subsets containing vertex 0.
													//If the graph is not vertex transitive, this should change to binom_h(N, K).

	printf("Checking %ld subsets.\n", number_of_subsets);	//Debugging, could be deleted.
	long offset;

	long size_of_subsets = BATCH*sizeof( short );	//Amount of memory that will be allocated for h_Results and d_Results.
	long size_of_A = N*N*sizeof(short);				//Amount of memory allocated to store the adjacency matrix.

	int i, j, k;	//indices
	h_A = (short *) malloc( size_of_A );				//allocate memory on host (CPU) for adjacency matrix
	h_Results = (short *) malloc( size_of_subsets );	//allocate memory on host (CPU) for results (indicator array)
	

	//Read in the adjacency matrix from the file.
	ifstream fin;
	fin.open(fileName);
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			fin >> h_A[i*N + j];	//Again, storing in 1D form.  Entry ij of an m x n matrix appears at entry i*n + j.
		}
	}
	fin.close();
	

	for(i=0; i<BATCH; i++){
		h_Results[i]=0;				//Initialize the results array (CPU-side).
	}


	cudaSetDevice( GPU );								//Select which GPU to use.
	cudaMalloc((void **) &d_A, size_of_A);				//allocate memory for the adjacency matrix on the GPU
	cudaMalloc((void **) &d_Results, size_of_subsets);	//allocate memory for the results (indicator array) on the GPU


	cudaMemcpy(d_A, h_A, size_of_A, cudaMemcpyHostToDevice);	//Copy the adjacency matrix from the CPU to the GPU
	cudaMemcpy(d_Results, h_Results, size_of_subsets, cudaMemcpyHostToDevice);	//Copy the Results (only initialized to 0) to the GPU.
	printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));		//Check for errors--debugging only (could be deleted).


	for(offset=0; offset<=number_of_subsets; offset+=BATCH){
		printf("Examining subsets %ld to %ld.\n", offset, offset+BATCH-1);
		
		//Invoke the kernel that inspects subsets.
		//IMPORTANT NOTE: If BATCH size is changed, it is important that examine_subsets<<< a, b>>>(...) satisfy a*b=BATCH
		//and keep in mind that BATCH/1000 is integer arithmetic (not floating point).
		//Depending on GPU architecture, it may provide a performance boost to make BATCH a power of 2 and then a=BATCH/b where b = 16, 32, or 64.
		examine_subsets<<< BATCH/1000, 1000 >>>(N, K, number_of_subsets, d_A, d_Results, offset);
		
		printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));		//Check for errors--debugging only (could be deleted).
																				//This only catches kernel invocation errors since it occurs before the cudaDeviceSynchronize().

		cudaMemcpy(h_Results, d_Results, size_of_subsets, cudaMemcpyDeviceToHost);	//Copy results indicator from GPU back to CPU.
		cudaDeviceSynchronize();	//Prevents the CPU code from moving on to the next for-loop before the execution on the GPU (and subsequent memory copy) finishes.

		for(i=0; (i<BATCH) && (i+offset<number_of_subsets) ; i++){
			if( h_Results[i] == 1 ){	//Found a potential switching set.
										//Print off the candidate switching set.
				for(j=0; j<K; j++){
					Kset[j]=0;
				}
				unrank_combination_h(N, K, i+offset, Kset);
				printf("Possible switching set %ld: ", i+offset);
				for(j=0; j<K; j++){
					unrank_combination_h(NN, KK, Kset[j], sKset);
					printf("{ ");
					for(k=0; k<KK; k++){
						printf("%d ", sKset[k]);
					}
					printf("} ");
				}
				printf("\n");
			}
		}
	}
	printf("Done.\n");

	free(h_A);
	free(h_Results);
	cudaFree(d_A);
	cudaFree(d_Results);
	
	return 0;
}
