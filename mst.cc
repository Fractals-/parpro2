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

MPI_Datatype MPI_Element;

double start_time;

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
  int i, curnode = 0, col;
  std::vector<int> queue; // Queue used to perform the BFS

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
      for ( i = row_ptr_begin[curnode]; i <= row_ptr_end[curnode]; i++ ) {
        col = col_ind[i];
        if ( component_id[col][0] == -1 ) {
          // Add the node 'col' to this disconnected graph
          queue.push_back(col);
          component_id[col][0] = num_graphs;
          component_id[col][1] = component_id[curnode][1] + 1;
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

  for ( i = 0; i <= max_BFS_lvl; i++ )
    fprintf(stderr, "%d: %d: level %d: %d: %d\n", rank, graph, i, new_lvl[i], lvl_size[i]);

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
  int node, source, tnode, cur_id = 0;
  unsigned int i, j;
  bool found, correct;

  for ( min_row = 0; min_row < n_rows; min_row++ ) {
    if ( component_id[min_row][0] == graph && component_id[min_row][1] == rank )
      break;
  }

  // As long as there is node that is not yet in a component continue
  while ( min_row < n_rows ) {
    //fprintf(stderr, "gen %d: %d: %d: %.2f\n", rank, min_row, cur_id, MPI_Wtime() - start_time);
    Component cur_comp(cur_id, min_row);
    component_id[min_row][2] = cur_id;
    cur_comp.nodes.push_back(min_row);

    found = cur_comp.findNextNode(node, source);

    // While this component can be expanded
    while ( found && component_id[node][1] == rank ) {
      if ( component_id[node][2] == -1 ) { // Add a single node to the component
        component_id[node][2] = cur_id;
        cur_comp.addNode(source, node);
        cur_comp.nodes.push_back(node);
      }
      else { // Merge with a previously finished component
        //fprintf(stderr, "comp %d: %d: %d: %d: %d: %.2f\n", rank, node, component_id[node][2], component_id[node][1], component_id[node][0], MPI_Wtime() - start_time);
        correct = false;

        for ( i = 0; i < finished_components.size(); i++ ) {
          if ( finished_components[i].id == component_id[node][2] ) {
            for ( j = 0; j < finished_components[i].nodes.size(); j++ ) {
              tnode = finished_components[i].nodes[j];
              component_id[tnode][2] = cur_id;
              cur_comp.nodes.push_back(tnode);
            }
            //fprintf(stderr, "reached1 %d: %d: %d: %.2f\n", rank, min_row, cur_id, MPI_Wtime() - start_time);
            // Merge the components
            cur_comp.addComponent(finished_components[i], node, source);
            //fprintf(stderr, "reached2 %d: %d: %d: %.2f\n", rank, min_row, cur_id, MPI_Wtime() - start_time);
            finished_components.erase(finished_components.begin() + i);
            correct = true;
            break;
          }
        }
        if ( !correct ){
          for ( i = 0; i < cur_comp.elements.size(); i++ )
            fprintf(stderr, "wha  %.2f: %d: %d\n", cur_comp.elements[i].dist, cur_comp.elements[i].col, cur_comp.elements[i].from);

          MPI_Finalize();
          abort();
        }
      }

      found = cur_comp.findNextNode(node, source);
    }
    //fprintf(stderr, "reach %d: %d: %d: %.2f\n", rank, min_row, cur_id, MPI_Wtime() - start_time);
    // Current component can no longer be expanded using available nodes
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
  unsigned int sizes[3];
  sizes[0] = comp.elements.size();
  sizes[1] = comp.edges_source.size();
  sizes[2] = comp.nodes.size();

  MPI_Send(&sizes, 3, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD);
  MPI_Send(&comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD);

  MPI_Send(&comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD);
  MPI_Send(&comp.edges_source[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD);
  MPI_Send(&comp.edges_target[0], sizes[1], MPI_INT, target_rank, 5, MPI_COMM_WORLD);

  MPI_Send(&comp.nodes[0], sizes[2], MPI_INT, target_rank, 6, MPI_COMM_WORLD);
}


// *************************************************************************************

/* Receives a single component from processor 'target_rank'
 * Parameters:
 *    cur_id      - The id for the received component
 *    target_rank - The source processor
 * Returns - The received component
 */
Component receiveComponent( int cur_id, int target_rank ){
  unsigned int sizes[3], i;
  MPI_Status status;
  MPI_Recv(&sizes, 3, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD, &status);

  Component new_comp(cur_id);
  new_comp.elements.resize(sizes[0]);
  new_comp.edges_source.resize(sizes[1]);
  new_comp.edges_target.resize(sizes[1]);

  MPI_Recv(&new_comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD, &status);

  MPI_Recv(&new_comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD, &status);
  MPI_Recv(&new_comp.edges_source[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD, &status);
  MPI_Recv(&new_comp.edges_target[0], sizes[1], MPI_INT, target_rank, 5, MPI_COMM_WORLD, &status);

  new_comp.nodes.resize(sizes[2]);
  MPI_Recv(&new_comp.nodes[0], sizes[2], MPI_INT, target_rank, 6, MPI_COMM_WORLD, &status);

  // Update data structure
  for ( i = 0; i < sizes[2]; i++ ) {
    component_id[new_comp.nodes[i]][1] = rank;
    component_id[new_comp.nodes[i]][2] = cur_id;
  }

  return new_comp;
}

// *************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////
// void debugComponents( std::vector<Component> finished_components ){
//   // DEBUG THE COMMUNICATION HERE
//   unsigned int i, j;
//   fprintf(stderr, "\n---------------\nProcessor %d:\n", rank);
//   for ( i = 0; i < finished_components.size(); i++ ) {
//     Component comp = finished_components[i];
//     fprintf(stderr, "\nComponent %d:\n", comp.id);
//     fprintf(stderr, "%d, %d: id = %d\n", rank, i, comp.id);
//     fprintf(stderr, "%d, %d: weight = %.1f\n", rank, i, comp.weight);
//     for ( j = 0; j < comp.elements.size(); j++ ){
//       Element el = comp.elements[j];
//       fprintf(stderr, "%d, %d: edge: %.1f, %d, %d\n", rank, i, el.dist, el.col, el.from);
//     }
//     fprintf(stderr, "\n");
//     for ( j = 0; j < comp.edges_source.size(); j++ ){
//       fprintf(stderr, "%d, %d: path: %d, %d\n", rank, i, comp.edges_source[j], comp.edges_target[j]);
//     }
//   }
// }

// *************************************************************************************

/* Combine the components from this 'subgraph' as much as possible
 * Parameters:
 *    finished_components - The components of this 'subgraph'
 */
void combineComponents( std::vector<Component>& finished_components ){
  int node, source;
  unsigned int i = 0, j, k;
  bool found;

  // For each component, attempt to merge it with another
  while ( i < finished_components.size() ) {
    Component cur_comp = finished_components[i];
    found = cur_comp.findNextNode(node, source);

    // While this component can be expanded
    while ( found && component_id[node][1] == rank ) {
      for ( k = 0; k < finished_components.size(); k++ ) {
        if ( finished_components[k].id == component_id[node][2] ) {
          for ( j = 0; j < finished_components[k].nodes.size(); j++ ) {
            component_id[finished_components[k].nodes[j]][2] = cur_comp.id;
            cur_comp.nodes.push_back(finished_components[k].nodes[j]);
          }
          // Merge the components
          cur_comp.addComponent(finished_components[k], node, source);
          finished_components.erase(finished_components.begin() + k);
          if ( k < i ) // Adjust i because of removal
            i--;
          break;
        }
      }
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
  fprintf(stderr, "reached1 %d: %d\n", rank, graph);
  int nstep, mod_rank,
      cur_id = finished_components[((int) finished_components.size() - 1)].id + 1;
  unsigned int num_comps, i;
  MPI_Status status;
  fprintf(stderr, "reached2 %d: %d\n", rank, graph);

  // Perform stepwise reduction
  while ( step != mpi_size ){
    nstep = 2 * step;
    mod_rank = rank % nstep;
    fprintf(stderr, "reached3 %d: %d\n", rank, graph);

    // Allow each processor to first finish creating its components
    // fprintf(stderr, "finished this part %.2f", MPI_Wtime());
    //MPI_Barrier(MPI_COMM_WORLD);
    // fprintf(stderr, "finished this part %.2f", MPI_Wtime());

    fprintf(stderr, "reached4 %d: %d\n", rank, graph);
    
    if ( mod_rank == 0 ){
      // Receive components from 'rank + step' and integrate them
      MPI_Recv(&num_comps, 1, MPI_UNSIGNED, rank + step, 0, MPI_COMM_WORLD, &status);

      for ( i = 0; i < num_comps; i++ ){
        Component new_comp = receiveComponent( cur_id, rank + step );
        finished_components.push_back(new_comp);
        cur_id++;
      }

      // Integrate/combine the components
      combineComponents(finished_components);
    }
    else if ( mod_rank - step == 0 ){
      // Send components to 'rank - step'
      num_comps = finished_components.size();
      fprintf(stderr, "num comp %d: %d: %d\n", rank, graph, (int) num_comps);
      MPI_Send(&num_comps, 1, MPI_UNSIGNED, rank - step, 0, MPI_COMM_WORLD);

      for ( i = 0; i < num_comps; i++ )
        sendComponent( finished_components[i], rank - step );
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

  fprintf(stderr, "gen %d: %d: %.2f\n", rank, graph, MPI_Wtime() - start_time);

  // Combine the results of the various processors
  mergeLevels(finished_components);

  fprintf(stderr, "merged %d: %d: %.2f\n", rank, graph, MPI_Wtime() - start_time);

  if ( finished_components.size() > 0 ){
    Component cur_comp = finished_components[0];
    finished_components.clear();
    return cur_comp;
  }
  else
    return NULL;
}

// *************************************************************************************

void outputMST( double elapsed_time, std::vector<Component>& finished_mst ){
  unsigned int i, j;
  fprintf(stdout, "\nElapsed time %.4f\n---------------\n", elapsed_time);
  for ( i = 0; i < finished_mst.size(); i++ ) {
    Component comp = finished_mst[i];
    fprintf(stdout, "\nMST %d:\n", i);
    fprintf(stdout, "weight = %.4f\n", comp.weight);
    for ( j = 0; j < comp.edges_source.size(); j++ ){
      fprintf(stdout, "%d, %d\n", comp.edges_source[j], comp.edges_target[j]);
    }
  }
  fprintf(stdout, "\n---------------\nElapsed time %.4f\n", elapsed_time);
}

// *************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////

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

  // Initialize MPI related matters
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  createElementType();

  start_time = MPI_Wtime();

  // Initialize component id's for each row to -1
  for ( int i = 0; i < n_rows; i++ )
    for ( int j = 0; j < 3; j++ )
      component_id[i][j] = -1;

  // Determine all disconnected graphs
  std::vector<int> graph_sizes; // Stores the sizes of all disconnected graphs
  std::vector<int> max_BFS_lvl; // Stores the maximum depth of BFS starting at the 'lowest' node
  determineGraphs(n_rows, graph_sizes, max_BFS_lvl);

  // // Debug output
  // if ( rank == 0 ){
  //   for ( int i = 0; i < n_rows; i++ )
  //     fprintf(stderr, "%d, %d, %d\n", component_id[i][0], component_id[i][1], component_id[i][2]);
  // }

  // Determine processor distribution for each graph and compute the MST
  std::vector<Component> finished_mst;
  for ( graph = 0; graph < num_graphs; graph++ ){
    // fprintf(stderr, "Graph %d: %d: %d: %d\n", graph, rank, graph_sizes[graph], num_graphs);
    if ( graph_sizes[graph] > 1000 ) {// Otherwise parallelization is unlikely to be helpful
      determineComponents(n_rows, graph_sizes[graph], max_BFS_lvl[graph]);
      // fprintf(stderr, "time %d: %d: %.2f\n", graph, rank, MPI_Wtime() - start_time);
      Component comp = generateMst(n_rows);
      if ( rank == 0 ){
        // fprintf(stderr, "Graph %d: %d: %d\n", graph, rank, (int) comp.nodes.size());
        finished_mst.push_back(comp);
      }
    }
    else if ( graph_sizes[graph] > 1 ) { // No mst exist for graphs of size 1
      for ( int i = 0; i < max_n_rows; i++ ){
        if ( component_id[i][0] == graph )
          component_id[i][1] = 0;
      }
      int temp_size = mpi_size;
      mpi_size = 1; // Ignore any parallelization
      if ( rank == 0 ) {
        Component comp = generateMst(n_rows);
        finished_mst.push_back(comp);
      }
      mpi_size = temp_size; 
    }
  }

  // Write mst's to file
  if ( rank == 0 ){
    // Compute elapsed time
    double elapsed_time = MPI_Wtime() - start_time;
    outputMST(elapsed_time, finished_mst);
  }

  MPI_Type_free(&MPI_Element);
  MPI_Finalize();

  return 0;
}
