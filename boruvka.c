#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

//Struct for storing edges
typedef struct edge{
	int a, b, wt;
}Edge;

//Finding representative for set in union find algorithm
int find(int a, int* parent){
	if(parent[a] != a)
		parent[a] = find(parent[a], parent);
	return parent[a];
}

//Merging two sets in union find algorithm
void Union(int a, int b, int* parent, int* rank){
	int tree1 = find(a, parent);
	int tree2 = find(b, parent);

	if(rank[tree1] < rank[tree2])
		parent[tree1] = tree2;
	else if(rank[tree1] > rank[tree2])
		parent[tree2] = tree1;
	else{
		parent[tree2] = tree1;
		rank[tree2]++;
	}
}


int main()
{
	MPI_Init(NULL, NULL);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int V, E, a, b, wt, i, j, edges_in_process, numTrees, MSTwt = 0;
	Edge *edges;

	//Scanning edges and edge partitioning
	if(world_rank==0){

		scanf("%d %d", &V, &E);
		edges_in_process = E/world_size + (int)(world_rank < (E % world_size));
		numTrees = V;

		MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);

		edges = (Edge*)malloc(edges_in_process * sizeof(Edge));
		j = 0;

		for(i = 0 ; i < E ; i++){

			scanf("%d %d %d", &a, &b, &wt);
			if(i % world_size == 0){
				edges[j].a = a;
				edges[j].b = b;
				edges[j].wt = wt;
				j++;
 			}
			else{
				MPI_Send(&a, 1, MPI_INT, i%world_size, 0, MPI_COMM_WORLD);
				MPI_Send(&b, 1, MPI_INT, i%world_size, 0, MPI_COMM_WORLD);
				MPI_Send(&wt, 1, MPI_INT, i%world_size, 0, MPI_COMM_WORLD);
			}
		}

		printf("\nNumber of edges are %d\n\nEdges in MST are:\n", V-1);
	}
	else{
		MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);

		numTrees = V;
		edges_in_process = E/world_size + (world_rank < (E % world_size));
		edges = (Edge*)malloc(edges_in_process * sizeof(Edge));

		for(i = 0 ; i < edges_in_process ; i++){
			MPI_Recv(&a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&wt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			edges[i].a = a;
			edges[i].b = b;
			edges[i].wt = wt;
		}
	}

	int* parent = (int*)malloc((V+1) * sizeof(int));
	int* rank = (int*)malloc((V+1) * sizeof(int));
	Edge* cheapest_Edge = (Edge*)malloc((V+1) * sizeof(Edge));

	for(i = 1 ; i <= V ; i++){
		parent[i] = i;
		rank[i] = 1;
	}

	//Keep combining trees until all we get single MST.
	while(numTrees > 1){
		for(i = 1 ; i <= V ; i++)
			cheapest_Edge[i].wt = INT_MAX;

		//Traverse through all edges and update cheapest of every component
		for(i = 0 ; i < edges_in_process ; i++){
			int tree1 = find(edges[i].a, parent);
			int tree2 = find(edges[i].b, parent);

			if(tree1 == tree2)
				continue;

			if(cheapest_Edge[tree1].wt > edges[i].wt){
				cheapest_Edge[tree1].a = edges[i].a;
				cheapest_Edge[tree1].b = edges[i].b;
				cheapest_Edge[tree1].wt = edges[i].wt;
			}

			if(cheapest_Edge[tree2].wt > edges[i].wt){
				cheapest_Edge[tree2].a = edges[i].a;
				cheapest_Edge[tree2].b = edges[i].b;
				cheapest_Edge[tree2].wt = edges[i].wt;
			}
		}

		//Send information regarding cheapest edge to master process
		if(world_rank != 0){
			for(i = 1 ; i <= V ; i++)
				if(cheapest_Edge[i].wt != INT_MAX){
					MPI_Send(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					a = cheapest_Edge[i].a;
					b = cheapest_Edge[i].b;
					wt = cheapest_Edge[i].wt;
					MPI_Send(&a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					MPI_Send(&b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					MPI_Send(&wt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				}
			i = -1;
			MPI_Send(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

			for(i = 1 ; i <= V ; i++)
				MPI_Bcast(&parent[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&numTrees, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}
		else{
			for(i = 1 ; i < world_size ; i++){
				MPI_Recv(&j, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				while(j != -1){
					MPI_Recv(&a, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&b, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&wt, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					if(cheapest_Edge[j].wt > wt){
						cheapest_Edge[j].a = a;
						cheapest_Edge[j].b = b;
						cheapest_Edge[j].wt = wt;
					}
					MPI_Recv(&j, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}

			//Get the cheapest edges and add them to MST
			for(i = 1 ; i <= V ; i++){
				if(cheapest_Edge[i].wt != INT_MAX){
					int tree1 = find(cheapest_Edge[i].a, parent);
					int tree2 = find(cheapest_Edge[i].b, parent);

					if(tree1 == tree2)
						continue;

					MSTwt += cheapest_Edge[i].wt;
					printf("%d %d\n", cheapest_Edge[i].a, cheapest_Edge[i].b);
					Union(tree1, tree2, parent, rank);
					numTrees--;
				}
			}

			for(i = 1 ; i <= V ; i++)
				MPI_Bcast(&parent[i], 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&numTrees, 1, MPI_INT, 0, MPI_COMM_WORLD);

		}
	}

	if(world_rank == 0)
		printf("\nWeight of MST is %d\n", MSTwt);

	MPI_Finalize();
	return 0;
}
