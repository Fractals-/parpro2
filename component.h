/* component.h
 * Assignment 2, Parallel Programming, LIACS
 *
 * Hanjo Boekhout - s1281348
 * 31-5-2016
 */

#ifndef __COMPONENT_H__
#define __COMPONENT_H__

extern double values[max_n_elements];

extern int col_ind[max_n_elements];
extern int row_ptr_begin[max_n_rows];
extern int row_ptr_end[max_n_rows];

extern int component_id[max_n_rows][3];
extern int graph; // The 'disconnected' graph currently being evaluated
extern int rank; // The rank of the process

struct Element {
  double dist; // Distance/weight between nodes
  int col;     // Node outside of the component
  int from;    // The node in the component from which this edge actually runs
};

class Component {
public:
  int id;          // Component id
  double weight;   // MST weight
  std::vector<Element> elements; // Connections from component
  std::vector<int> nodes;        // Indicates which nodes are in this component

  // Constructors
  Component( int tid );
  Component( int tid, int row_idx );
  ~Component();

  /* Finds the closest node 'node' to the component
   * Parameters:
   *    node   - The target node closest to the component
   *    source - The source node to the closest node
   * Returns - True if a node was found, False if the component is complete
   */
  bool findNextNode( int &node, int &source);

  /* Adds the closest node 'node' to the component
   * Parameters:
   *    node   - The target node closest to the component
   */
  void addNode( int node );

  /* Adds the closest component 'comp' to the component given by the connection
   * between to 'node'
   * Parameters:
   *    node   - The target node closest to the component
   */
  void addComponent( Component &comp, int node );
};

#endif /* __COMPONENT_H__ */