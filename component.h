#ifndef __COMPONENT_H__
#define __COMPONENT_H__

extern double values[max_n_elements];

extern int col_ind[max_n_elements];
extern int row_ptr_begin[max_n_rows];
extern int row_ptr_end[max_n_rows];

extern int component_id[max_n_rows][3];
extern int graph; // The 'disconnected' graph currently being evaluated
extern int rank; // The rank of the process

struct Element{
  double dist; // Distance/weight between nodes
  int col;     // Node outside of the component
  int from;    // The node in the component from which this edge actually runs
};

class Component {
public:
  int id;          // Component id
  double weight;   // MST weight
  std::vector<Element> elements; // Connections from component
//  std::vector<bool> nodes;       // Indicates which nodes are in this component
  std::vector<int> edges_source; // source + target = edge in MST
  std::vector<int> edges_target;
  std::vector<int> nodes;

  Component( int tid );
  Component( int tid, int row_idx );
  Component( int tid, Component &comp );
  ~Component();

  bool findNextNode( int &node, int &source);

  void addNode( int source, int node );
  void addComponent( Component &comp, int node, int source );
};

#endif /* __COMPONENT_H__ */