#include "general.h"
#include "component.h"

Component::Component( int tid ){
  id = tid;
}

// *************************************************************************************

Component::Component( int tid, int row_idx ){
  id = tid;
  weight = 0.0;

  Element el;
  int col;

  for ( int i = row_ptr_begin[row_idx]; i <= row_ptr_end[row_idx]; i++ ) {
    col = col_ind[i];
    // Target node is in the correct graph (and not a self edge)
    if ( component_id[col][0] == graph && col != row_idx ) {
      el.dist = values[i];
      el.col = col_ind[i];
      el.from = row_idx;
      elements.push_back(el);
    }
  }
}

// *************************************************************************************

Component::~Component(){
  elements.clear();
  edges_source.clear();
  edges_target.clear();
  nodes.clear();
}

// *************************************************************************************

bool Component::findNextNode( int &node, int &source ){
  unsigned int i;
  double min_value = DBL_MAX;
  node = -1;
  Element el;

  for ( i = 0; i < elements.size(); i++ ) {
    el = elements[i];
    if ( el.dist < min_value ){
      min_value = el.dist;
      node = el.col;
      source = el.from;
    }
  }

  fprintf(stderr, "%d: %d: %d\n", (int) elements.size(), node, id);
  // Return false if the component can not be further expanded
  if ( node >= 0 )
    return true;
  else
    return false;
}

// *************************************************************************************

void Component::addNode( int source, int node ){
  // Update mst
  edges_source.push_back(source);
  edges_target.push_back(node);

  unsigned int i = 0;
  int j = row_ptr_begin[node],
      col = col_ind[j];
  Element el, nel;
  // Create combined 'out-edge list'
  while ( i < elements.size() && j <= row_ptr_end[node]) {
    el = elements[i];

    if ( el.col == node ){ // Remove newly added edge (prevent self edge)
      weight += el.dist;
      elements.erase(elements.begin() + i);
    }
    else if ( col < el.col ) { // Insert edge to a 'new' vertex
      if ( component_id[col][0] == graph &&  // Correct graph
           !( component_id[col][1] == rank && component_id[col][2] == id ) ) { // No self edge
        nel.dist = values[j];
        nel.col = col;
        nel.from = node;
        elements.insert(elements.begin() + i, nel);
        i++;
      }
      j++;
    }
    else if ( col == el.col ){
      if ( values[j] < el.dist ){
        el.dist = values[j];
        el.col = col;
        el.from = node;
        elements[i] = el;
      }
      i++;
      j++;
    }
    else {
      i++;
    }

    col = col_ind[j];
  }

  // Add any remaining elements at the end
  while ( j <= row_ptr_end[node] ) {
    if ( component_id[col][0] == graph &&  // Correct graph
         !( component_id[col][1] == rank && component_id[col][2] == id ) ) { // No self edge
      nel.dist = values[j];
      nel.col = col;
      nel.from = node;
      elements.push_back(nel);
    }
    j++;
    col = col_ind[j];
  }
}

// *************************************************************************************

void Component::addComponent( Component &comp, int node, int source ){
  unsigned int i, j = 0;

  weight += comp.weight;
  // Update mst
  for ( i = 0; i < comp.edges_source.size(); i++ ) {
    edges_source.push_back(comp.edges_source[i]);
    edges_target.push_back(comp.edges_target[i]);
  }
  edges_source.push_back(source);
  edges_target.push_back(node);

  i = 0;
  Element el, el2;
  // Create combined 'out-edge list'
  while ( i < elements.size() && j < comp.elements.size() ) {
    el = elements[i];
    el2 = comp.elements[j];

    if ( el.col == node ){ // Remove newly added edge (prevent self edge)
      weight += el.dist;
      elements.erase(elements.begin() + i);
    }
    else if ( el2.col < el.col ) { // Insert edge to a 'new' vertex
      if ( component_id[el2.col][0] == graph &&  // Correct graph
           !( component_id[el2.col][1] == rank && component_id[el2.col][2] == id ) ) { // No self edge
        elements.insert(elements.begin() + i, el2);
        i++;
      }
      j++;
    }
    else if ( el2.col == el.col ){
      if ( el2.dist < el.dist )
        elements[i] = el2;
      i++;
      j++;
    }
    else {
      if ( component_id[el.col][0] == graph &&  // Correct graph
          !( component_id[el.col][1] == rank && component_id[el.col][2] == id ) ) // No self edge
        i++;
      else // Remove self edge
        elements.erase(elements.begin() + i);
    }
  }

  // Add any remaining elements at the end
  while ( j < comp.elements.size() ) {
    el2 = comp.elements[j];
    if ( component_id[el2.col][0] == graph &&  // Correct graph
         !( component_id[el2.col][1] == rank && component_id[el2.col][2] == id ) ) { // No self edge
      elements.push_back(el2);
    }
    j++;
  }
}