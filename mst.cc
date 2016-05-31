/* mst.cc
 * Assignment 2, Parallel Programming, LIACS
 *
 * Hanjo Boekhout - s1281348
 * 31-5-2016
 */

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <cmath>
#include "mpi.h"

#include "general.h"
#include "matrix.h"
#include "component.h"

/* Global variables holding the matrix data.
 * And global variables holding mpi information.
 */

double values[max_n_elements];

int col_ind[max_n_elements];
int row_ptr_begin[max_n_rows];
int row_ptr_end[max_n_rows];

int component_id[max_n_rows][3];
int graph; // The 'disconnected' graph currently being evaluated
int rank; // The rank of the process

static int mpi_size; // The number of processes
static int min_row; // The floor for the lowest row not yet in a component
static int num_graphs; // The number of disconnected graphs

static int component_position[max_n_rows];

MPI_Datatype MPI_Element;

int path[max_n_rows][2];
int path_index;

// *************************************************************************************

/* Creates the 'MPI_Element' mpi datatype, such that Elements can be communicated
 */
void createElementType(){
  const int    nitems = 3;
  int          blocklengths[3] = {1, 1, 1};
  MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
  MPI_Datatype MPI_Element_proto;
  MPI_Aint     offsets[3];

  offsets[0] = offsetof(Element, dist);
  offsets[1] = offsetof(Element, col);
  offsets[2] = offsetof(Element, from);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_Element_proto);

  // For insurance we resize the constructed type
  MPI_Aint lb, extent;
  MPI_Type_get_extent(MPI_Element_proto, &lb, &extent);
  extent = sizeof(Element);
  MPI_Type_create_resized(MPI_Element_proto, lb, extent, &MPI_Element);

  MPI_Type_commit(&MPI_Element);
}

// *************************************************************************************

/* Determine all disconnected graphs, their sizes and depth (for BFS from 'lowest' node)
 * Parameters:
 *    n_rows       - The number of rows in the matrix
 *    graph_sizes  - The sizes of all disconnected graphs
 *    max_BFS_lvl  - The maximum depth of a BFS starting at the 'lowest' node
 */
void determineGraphs( int n_rows, std::vector<int>& graph_sizes, std::vector<int>& max_BFS_lvl ){
  int i, curnode = 0, col, lvl;
  std::vector<int> queue; // Queue used to perform the BFS
  queue.reserve(1000000);

  min_row = 0; // Stores 'lowest' node that is not in a graph yet
  num_graphs = -1;

  // For each disconnected graph
  while ( min_row < n_rows ) {
    // Initialize new disconnected graph
    num_graphs++;
    queue.push_back(min_row);
    component_id[min_row][0] = num_graphs;
    component_id[min_row][1] = 0;
    graph_sizes.push_back(1);

    // Do a BFS search
    while ( !queue.empty() ){
      curnode = queue[0];
      lvl = component_id[curnode][1];
      for ( i = row_ptr_begin[curnode]; i <= row_ptr_end[curnode]; i++ ) {
        col = col_ind[i];
        if ( component_id[col][0] == -1 ) {
          // Add the node 'col' to this disconnected graph
          queue.push_back(col);
          component_id[col][0] = num_graphs;
          component_id[col][1] = lvl + 1;
          graph_sizes[num_graphs]++;
        }
      }
      queue.erase(queue.begin());
    }
    max_BFS_lvl.push_back(component_id[curnode][1]);

    //Determine start of next disconnected graph
    for ( ; min_row < n_rows; min_row++ ) {
      if ( component_id[min_row][0] == -1 ) // A node was not yet added to a graph
        break;
    }
  }

  num_graphs++;
}

// *************************************************************************************

/* Set each BFS level to a processor
 * Parameters:
 *    n_rows      - The number of rows in the matrix
 *    graph_size  - The size of a disconnected graph
 *    max_BFS_lvl - The maximum depth of a BFS starting at the 'lowest' node
 */
void determineComponents( int n_rows, int graph_size, int max_BFS_lvl ){
  int i, proc_size, curlvl,
      maxsize = graph_size / mpi_size; // The maximum number of nodes per processor
  int new_lvl[max_BFS_lvl + 1];  // Stores for each BFS lvl it's assigned processor
  int lvl_size[max_BFS_lvl + 1]; // Stores the size of each BFS lvl

  for ( i = 0; i <= max_BFS_lvl; i++ )
    lvl_size[i] = 0;
  for ( i = 0; i < n_rows; i++ ){
    if ( graph == component_id[i][0] )
      lvl_size[component_id[i][1]]++;
  }

  // Determine which BFS levels are assigned to which processor
  curlvl = 0;
  for ( i = 0; i < mpi_size; i++ ) {
    new_lvl[curlvl] = i; // At least one BFS_lvl per processor
    proc_size = lvl_size[curlvl];

    while ( proc_size < maxsize && mpi_size - (i + 1) < max_BFS_lvl - curlvl ) {
      curlvl++;
      new_lvl[curlvl] = i;
      proc_size += lvl_size[curlvl];
    }
    curlvl++;
  }

  // Set the processors
  for ( i = 0; i < n_rows; i++ )
    component_id[i][1] = new_lvl[component_id[i][1]];
}

// *************************************************************************************

/* Generate the components for a 'subgraph', taking into account edges leaving
 * this subgraph
 * Parameters:
 *    n_rows              - The number of rows in the matrix
 *    finished_components - The set of components of this 'subgraph'
 */
void generateComponents( int n_rows, std::vector<Component>& finished_components ){
  int node, source, cur_id = 0, i, index, idx;
  unsigned int j;
  bool found;

  for ( min_row = 0; min_row < n_rows; min_row++ ) {
    if ( component_id[min_row][0] == graph && component_id[min_row][1] == rank )
      break;
  }

  // As long as there is node that is not yet in a component continue
  while ( min_row < n_rows ) {
    Component cur_comp(cur_id, min_row);
    component_id[min_row][2] = cur_id;
    cur_comp.nodes.push_back(min_row);

    found = cur_comp.findNextNode(node, source);

    // While this component can be expanded
    while ( found && component_id[node][1] == rank ) {
      path[path_index][0] = source;
      path[path_index][1] = node;
      path_index++;
      if ( component_id[node][2] == -1 ) { // Add a single node to the component
        component_id[node][2] = cur_id;
        cur_comp.addNode(node);
        cur_comp.nodes.push_back(node);
      }
      else { // Merge with a previously finished component
        idx = component_id[node][2];
        index = component_position[idx];

        cur_comp.nodes.insert(cur_comp.nodes.end(), finished_components[index].nodes.begin(),
                              finished_components[index].nodes.end());
        for ( j = 0; j < finished_components[index].nodes.size(); j++ ) {
          component_id[finished_components[index].nodes[j]][2] = cur_id;
        }
        // Merge the components
        cur_comp.addComponent(finished_components[index], node);
        finished_components.erase(finished_components.begin() + index);

        for ( i = idx + 1; i < cur_id; i++ )
          component_position[i]--;
      }

      found = cur_comp.findNextNode(node, source);
    }
    // Current component can no longer be expanded using available nodes
    component_position[cur_id] = finished_components.size();
    finished_components.push_back(cur_comp);
    cur_id++;

    //Determine start of the next component
    for ( ; min_row < n_rows; min_row++ ) {
      if ( component_id[min_row][0] == graph && component_id[min_row][1] == rank &&
           component_id[min_row][2] == -1 )
        break;
    }
  }
}

// *************************************************************************************

/* Sends a single component to processor 'target_rank'
 * Parameters:
 *    comp        - The component to send
 *    target_rank - The target processor
 */
void sendComponent( Component& comp, int target_rank ){
  unsigned int sizes[2];
  sizes[0] = comp.elements.size();
  sizes[1] = comp.nodes.size();

  MPI_Send(&sizes, 2, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD);
  MPI_Send(&comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD);

  MPI_Send(&comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD);

  MPI_Send(&comp.nodes[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD);
}


// *************************************************************************************

/* Receives a single component from processor 'target_rank'
 * Parameters:
 *    cur_id      - The id for the received component
 *    target_rank - The source processor
 * Returns - The received component
 */
Component receiveComponent( int cur_id, int target_rank ){
  unsigned int sizes[2], i;
  MPI_Status status;
  MPI_Recv(&sizes, 2, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD, &status);

  Component new_comp(cur_id);
  new_comp.elements.resize(sizes[0]);

  MPI_Recv(&new_comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD, &status);
  MPI_Recv(&new_comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD, &status);

  new_comp.nodes.resize(sizes[1]);
  MPI_Recv(&new_comp.nodes[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD, &status);

  // Update data structure
  for ( i = 0; i < sizes[1]; i++ ) {
    component_id[new_comp.nodes[i]][1] = rank;
    component_id[new_comp.nodes[i]][2] = cur_id;
  }

  return new_comp;
}

// *************************************************************************************

/* Combine the components from this 'subgraph' as much as possible
 * Parameters:
 *    finished_components - The components of this 'subgraph'
 */
void combineComponents( std::vector<Component>& finished_components ){
  int node, source, max_id = 0, k, index, idx;
  unsigned int i = 0, j;
  bool found;

  if ( !finished_components.empty() )
    max_id = finished_components[ (int) finished_components.size() - 1].id;

  // For each component, attempt to merge it with another
  while ( i < finished_components.size() ) {
    Component cur_comp = finished_components[i];
    found = cur_comp.findNextNode(node, source);

    // While this component can be expanded
    while ( found && component_id[node][1] == rank ) {
      idx = component_id[node][2];
      index = component_position[idx];
      path[path_index][0] = source;
      path[path_index][1] = node;
      path_index++;

      cur_comp.nodes.insert(cur_comp.nodes.end(), finished_components[index].nodes.begin(),
                            finished_components[index].nodes.end());
      for ( j = 0; j < finished_components[index].nodes.size(); j++ ) {
        component_id[finished_components[index].nodes[j]][2] = cur_comp.id;
      }
      // Merge the components
      cur_comp.addComponent(finished_components[index], node);
      finished_components.erase(finished_components.begin() + index);

      for ( k = idx + 1; k <= max_id; k++ )
        component_position[k]--;

      if ( index < (int) i ) // Adjust i because of removal
        i--;

      found = cur_comp.findNextNode(node, source);
    }

    finished_components[i] = cur_comp;
    i++;
  }
}

// *************************************************************************************

/* Combines the components found for all processors into a single component
 * Parameters:
 *    finished_components - The components of this processor
 */
void mergeLevels( std::vector<Component>& finished_components ){
  int step = 1; // Stores the current step size
  int nstep, mod_rank, path_size,
      cur_id = -1;
  if ( !finished_components.empty() )
    cur_id = finished_components[((int) finished_components.size() - 1)].id + 1;
  unsigned int num_comps, i;
  MPI_Status status;

  // Perform stepwise reduction
  while ( step != mpi_size ){
    nstep = 2 * step;
    mod_rank = rank % nstep;

    if ( mod_rank == 0 ){
      // Receive components from 'rank + step' and integrate them
      MPI_Recv(&num_comps, 1, MPI_UNSIGNED, rank + step, 0, MPI_COMM_WORLD, &status);

      for ( i = 0; i < num_comps; i++ ){
        Component new_comp = receiveComponent( cur_id, rank + step );
        component_position[cur_id] = finished_components.size();
        finished_components.push_back(new_comp);
        cur_id++;
      }

      MPI_Recv(&path_size, 1, MPI_INT, rank + step, 5, MPI_COMM_WORLD, &status);
      MPI_Recv(&path[path_index][0], path_size * 2, MPI_INT, rank + step, 6, MPI_COMM_WORLD, &status);
      path_index += path_size;

      // Integrate/combine the components
      combineComponents(finished_components);

    }
    else if ( mod_rank - step == 0 ){
      // Send components to 'rank - step'
      num_comps = finished_components.size();
      MPI_Send(&num_comps, 1, MPI_UNSIGNED, rank - step, 0, MPI_COMM_WORLD);

      for ( i = 0; i < num_comps; i++ )
        sendComponent( finished_components[i], rank - step );

      MPI_Send(&path_index, 1, MPI_INT, rank - step, 5, MPI_COMM_WORLD);
      MPI_Send(&path[0][0], path_index * 2, MPI_INT, rank - step, 6, MPI_COMM_WORLD);
    }

    step = nstep;
  }
}

// *************************************************************************************

/* Generates the mst of a graph
 * Parameters:
 *    n_rows - The number of rows in the matrix
 * Returns - The component containing the mst
 */
Component generateMst( int n_rows ){
  std::vector<Component> finished_components;
  generateComponents(n_rows, finished_components);

  // Combine the results of the various processors
  mergeLevels(finished_components);

  if ( finished_components.size() > 0 ){
    Component cur_comp = finished_components[0];
    finished_components.clear();
    return cur_comp;
  }
  else
    return NULL;
}

// *************************************************************************************

/* Writes a minimum spanning tree to the standard out
 * Parameters:
 *    comp - The component containing the mst
 */
void outputMST( Component comp ){
  if ( rank == 0 ) {
    fprintf(stdout, "\nMST %d:\n", graph);
    fprintf(stdout, "weight = %.4f\n", comp.weight);
    // for (int i = 0; i < path_index; i++ ){
    //   fprintf(stdout, "%d, %d\n", path[i][0], path[i][1]);
    // }
    fprintf(stdout, "number_nodes = %d\n", (int) comp.nodes.size());
  }
}

// *************************************************************************************

int
main(int argc, char **argv)
{
  if (argc != 2)
    {
      fprintf(stderr, "usage: %s <filename>\n", argv[0]);
      return -1;
    }

  int nnz, n_rows, n_cols;
  bool ok(false);

  ok = load_matrix_market(argv[1], max_n_elements, max_n_rows,
                          nnz, n_rows, n_cols,
                          values, col_ind, row_ptr_begin, row_ptr_end);
  if (!ok)
    {
      fprintf(stderr, "failed to load matrix.\n");
      return -1;
    }

  // Initialize component id's for each row to -1
  for ( int i = 0; i < n_rows; i++ ){
    for ( int j = 0; j < 3; j++ )
      component_id[i][j] = -1;
    component_position[i] = -1;
  }

  // Initialize MPI related matters
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  createElementType();

  double start_time = MPI_Wtime();

  // Determine all disconnected graphs
  std::vector<int> graph_sizes; // Stores the sizes of all disconnected graphs
  std::vector<int> max_BFS_lvl; // Stores the maximum depth of BFS starting at the 'lowest' node
  determineGraphs(n_rows, graph_sizes, max_BFS_lvl);

  // Determine processor distribution for each graph and compute the MST
  std::vector<Component> finished_mst;
  for ( graph = 0; graph < num_graphs; graph++ ){
    path_index = 0;
    if ( graph_sizes[graph] > 1000 ) {// Otherwise parallelization is unlikely to be helpful
      determineComponents(n_rows, graph_sizes[graph], max_BFS_lvl[graph]);
      outputMST(generateMst(n_rows));
    }
    else if ( graph_sizes[graph] > 1 ) { // No mst exist for graphs of size 1
      for ( int i = 0; i < max_n_rows; i++ ){
        if ( component_id[i][0] == graph )
          component_id[i][1] = 0;
      }
      int temp_size = mpi_size;
      mpi_size = 1; // Ignore any parallelization
      if ( rank == 0 )
        outputMST(generateMst(n_rows));
      mpi_size = temp_size; 
    }
  }

  // Write elapsed time to file
  if ( rank == 0 ){
    // Compute elapsed time
    double elapsed_time = MPI_Wtime() - start_time;
    fprintf(stdout, "\n---------------\nElapsed time %.4f\n", elapsed_time);
  }

  MPI_Type_free(&MPI_Element);
  MPI_Finalize();

  return 0;
}
