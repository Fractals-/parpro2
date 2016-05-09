#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include "mpi.h"

#include "general.h"
#include "matrix.h"
#include "component.h"

/* Code taken from the GLIBC manual.
 *
 * Subtract the ‘struct timespec’ values X and Y,
 * storing the result in RESULT.
 * Return 1 if the difference is negative, otherwise 0.
 */
static int
timespec_subtract (struct timespec *result,
                   struct timespec *x,
                   struct timespec *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

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


// *************************************************************************************

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

// Determine all disconnected graphs
void determineGraphs( int n_rows, std::vector<int>& graph_sizes, std::vector<int>& max_BFS_lvl ){
  int i, curnode = 0, col;
  std::vector<int> queue;

  min_row = 0;
  num_graphs = -1;

  // For each disconnected graph
  while ( min_row < n_rows ) {
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
          queue.push_back(col);
          component_id[col][0] = num_graphs;
          component_id[col][1] = component_id[curnode][1] + 1;
          graph_sizes[num_graphs]++;
        }
      }
      queue.erase(queue.begin());
    }
    max_BFS_lvl.push_back(component_id[curnode][1]);

    //Determine start of next component
    for ( ; min_row < n_rows; min_row++ ) {
      if ( component_id[min_row][0] == -1 ) // A node was not yet added to a graph
        break;
    }
  }

  num_graphs++;
}

// *************************************************************************************

// Set each BFS level to a processor
void determineComponents( int n_rows, int graph_size, int max_BFS_lvl ){
  int i, proc_size, curlvl,
      maxsize = graph_size / mpi_size;
  int new_lvl[max_BFS_lvl + 1];
  int lvl_size[max_BFS_lvl + 1];
  for ( i = 0; i <= max_BFS_lvl; i++ )
    lvl_size[i] = 0;

  for ( i = 0; i < n_rows; i++ )
    lvl_size[component_id[i][1]]++;

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

// Generate the components for a subgraph
void generateComponents( int n_rows, std::vector<Component>& finished_components ){
  int node, source, cur_id = 0, j;
  unsigned int i;
  bool found;

  for ( min_row = 0; min_row < n_rows; min_row++ ) {
    if ( component_id[min_row][0] == graph && component_id[min_row][1] == rank )
      break;
  }

  // As long as there is node that is not yet in a component
  while ( min_row < n_rows ) {
    Component cur_comp(cur_id, min_row);
    component_id[min_row][2] = cur_id;

    found = cur_comp.findNextNode(node, source);

    while ( found && component_id[node][1] == rank ) {
      if ( component_id[node][2] == -1 ) { // Add a single node to the component
        // fprintf(stderr, "hallo %d, %d\n\n", node, source);
        component_id[node][2] = cur_id;
        cur_comp.addNode(source, node);
      }
      else { // Merge with a previously finished component
        for ( i = 0; i < finished_components.size(); i++ ) {
          if ( finished_components[i].id == component_id[node][2] ) {
            // Set all elements of finished component to be in the current one
            for ( j = 0; j < n_rows; j++ ) {
              if ( component_id[j][0] == graph && component_id[j][1] == rank &&
                   component_id[j][2] == finished_components[i].id )
                component_id[j][2] = cur_id;
            }
            // Merge the componentes
            cur_comp.addComponent(finished_components[i], node, source);
            finished_components.erase(finished_components.begin() + i);
            break;
          }
        }
      }

      found = cur_comp.findNextNode(node, source);
    }
    // Current component can no longer be expanded using available nodes
    finished_components.push_back(cur_comp);
    cur_id++;

    //Determine start of next component
    for ( ; min_row < n_rows; min_row++ ) {
      if ( component_id[min_row][0] == graph && component_id[min_row][1] == rank &&
           component_id[min_row][2] == -1 )
        break;
    }
  }
}

// *************************************************************************************

void sendComponent( Component& comp, int target_rank ){
  unsigned int sizes[2];
  sizes[0] = comp.elements.size();
  sizes[1] = comp.edges_source.size();

  MPI_Send(&sizes, 2, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD);
  MPI_Send(&comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD);

  MPI_Send(&comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD);
  MPI_Send(&comp.edges_source[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD);
  MPI_Send(&comp.edges_target[0], sizes[1], MPI_INT, target_rank, 5, MPI_COMM_WORLD);

  MPI_Send(&comp.id, 1, MPI_INT, target_rank, 6, MPI_COMM_WORLD)
}


// *************************************************************************************

Component receiveComponent( int n_rows, int cur_id, int target_rank ){
  unsigned int sizes[2];
  int id;
  MPI_Status status;
  MPI_Recv(&sizes, 2, MPI_UNSIGNED, target_rank, 1, MPI_COMM_WORLD, &status);

  Component new_comp(cur_id);
  new_comp.elements.resize(sizes[0]);
  new_comp.edges_source.resize(sizes[1]);
  new_comp.edges_target.resize(sizes[1]);

  MPI_Recv(&new_comp.weight, 1, MPI_DOUBLE, target_rank, 2, MPI_COMM_WORLD, &status);

  MPI_Recv(&new_comp.elements[0], sizes[0], MPI_Element, target_rank, 3, MPI_COMM_WORLD, &status);
  MPI_Recv(&new_comp.edges_source[0], sizes[1], MPI_INT, target_rank, 4, MPI_COMM_WORLD, &status);
  MPI_Recv(&new_comp.edges_target[0], sizes[1], MPI_INT, target_rank, 5, MPI_COMM_WORLD, &status);

  MPI_Recv(&id, 1, MPI_INT, target_rank, 6, MPI_COMM_WORLD, &status);

  for ( int i = 0; i < n_rows; i++ ){
    if ( component_id[i][0] == graph && component_id[i][1] == target_rank &&
         component_id[i][2] == id)
      component_id[i][2] = cur_id;
  }

  return new_comp;
}

// *************************************************************************************

void debugComponents( std::vector<Component> finished_components ){
  // DEBUG THE COMMUNICATION HERE
  unsigned int i, j;
  fprintf(stderr, "\n---------------\nProcessor %d:\n", rank);
  for ( i = 0; i < finished_components.size(); i++ ) {
    Component comp = finished_components[i];
    fprintf(stderr, "\nComponent %d:\n", i);
    fprintf(stderr, "%d, %d: id = %d\n", rank, i, comp.id);
    fprintf(stderr, "%d, %d: weight = %.1f\n", rank, i, comp.weight);
    for ( j = 0; j < comp.elements.size(); j++ ){
      Element el = comp.elements[j];
      fprintf(stderr, "%d, %d: edge: %.1f, %d, %d\n", rank, i, el.dist, el.col, el.from);
    }
    fprintf(stderr, "\n");
    for ( j = 0; j < comp.edges_source.size(); j++ ){
      fprintf(stderr, "%d, %d: path: %d, %d\n", rank, i, comp.edges_source[j], comp.edges_target[j]);
    }
  }
}

// *************************************************************************************

void mergeLevels( int n_rows, std::vector<Component>& finished_components ){
  int step = 1;
  int nstep, mod_rank,
      cur_id = finished_components[((int) finished_components.size() - 1)].id + 1;
  unsigned int num_comps, i;
  MPI_Status status;

  // Perform stepwise reduction
  while ( step != mpi_size ){
    nstep = 2 * step;
    mod_rank = rank % nstep;

    // debugComponents(finished_components);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if ( mod_rank == 0 ){
      // Receive components from 'rank + step' and integrate them
      MPI_Recv(&num_comps, 1, MPI_UNSIGNED, rank + step, 0, MPI_COMM_WORLD, &status);

      for ( i = 0; i < num_comps; i++ ){
        Component new_comp = receiveComponent( n_rows, cur_id, rank + step );
        finished_components.push_back(new_comp);
      }

      // debugComponents(finished_components);

    }
    else if ( mod_rank - step == 0 ){
      // Send components to 'rank - step'
      num_comps = finished_components.size();
      MPI_Send(&num_comps, 1, MPI_UNSIGNED, rank - step, 0, MPI_COMM_WORLD);

      for ( i = 0; i < num_comps; i++ )
        sendComponent( finished_components[i], rank - step );

    }

    step = nstep;
  }
}

// order of things:
//   - number of components
//   - sizes of components (can be done in one go)
//   - one by one each component

// *************************************************************************************

// Generate the mst of a graph
Component generateMst( int n_rows ){
  std::vector<Component> finished_components;
  generateComponents(n_rows, finished_components);

  // Combine the results of the various processors
  mergeLevels(n_rows, finished_components);


  if ( finished_components.size() > 0 ){
    Component cur_comp = finished_components[0];
    finished_components.clear();
    return cur_comp;
  }
  else
    return NULL;
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

  //dump_nonzeros(n_rows, values, col_ind, row_ptr_begin, row_ptr_end);
  //fprintf(stderr, "what\n");

  // Initialize component id's for each row to -1
  for ( int i = 0; i < n_rows; i++ )
    for ( int j = 0; j < 3; j++ )
      component_id[i][j] = -1;

  // Determine all disconnected graphs
  std::vector<int> graph_sizes;
  std::vector<int> max_BFS_lvl;
  determineGraphs(n_rows, graph_sizes, max_BFS_lvl);

  // Initialize MPI related matters
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  createElementType();

  // Debug output
  if ( rank == 0 ){
    for ( int i = 0; i < n_rows; i++ )
      fprintf(stderr, "%d, %d, %d\n", component_id[i][0], component_id[i][1], component_id[i][2]);
  }

  // Determine processor distribution for each graph and compute mst
  std::vector<Component> finished_mst;
  for ( graph = 0; graph < num_graphs; graph++ ){
    if ( graph_sizes[graph] > 3 ) {// Otherwise parallelization is unlikely to be helpful
      determineComponents(n_rows, graph_sizes[graph], max_BFS_lvl[graph]);
      Component comp = generateMst(n_rows);
      if ( rank == 0 )
        finished_mst.push_back(comp);
    }
    else{
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
    debugComponents(finished_mst);
    // TODO
  }


  // Debug output
  if ( rank == 0 ){
    for ( int i = 0; i < n_rows; i++ )
      fprintf(stderr, "%d, %d, %d\n", component_id[i][0], component_id[i][1], component_id[i][2]);
  }

  MPI_Type_free(&MPI_Element);
  MPI_Finalize();

  return 0;
}
