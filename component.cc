/* component.cc
 * Assignment 2, Parallel Programming, LIACS
 *
 * Hanjo Boekhout - s1281348
 * 31-5-2016
 */

#include "general.h"
#include "component.h"

Component::Component( int tid ){
  id = tid;
  weight = 0.0;
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
    if ( col != row_idx ) {
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
    if ( el.dist < min_value && component_id[el.col][2] != id  ){
      min_value = el.dist;
      node = el.col;
      source = el.from;
    }
  }

  // Return false if the component can not be further expanded
  if ( node >= 0 )
    return true;
  else
    return false;
}

// *************************************************************************************

void Component::addNode( int node ){
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
      if ( component_id[col][2] != id ) { // No self edge
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

  // Remove any remaining edges to the new node
  while ( i < elements.size() ) {
    if ( elements[i].col == node ){
      weight += elements[i].dist;
      elements.erase(elements.begin() + i);
      return;
    }
    i++;
  }

  // Add any remaining elements at the end
  while ( j <= row_ptr_end[node] ) {
    if ( component_id[col][2] != id ) { // No self edge
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

void Component::addComponent( Component &comp, int node ){
  unsigned int i, j = 0;

  std::vector<Element> temp = elements;
  elements.clear();
  elements.reserve(temp.size() + comp.elements.size());

  weight += comp.weight;

  i = 0;
  // Create combined 'out-edge list'
  while ( i < temp.size() && j < comp.elements.size() ) {
    if ( temp[i].col == node ){ // Remove newly added edge (prevent self edge)
      weight += temp[i].dist;
      i++;
    }
    else if ( comp.elements[j].col < temp[i].col ) { // Insert edge to a 'new' vertex
      if ( component_id[comp.elements[j].col][2] != id ) { // No self edge
        elements.push_back(comp.elements[j]);
      }
      j++;
    }
    else if ( comp.elements[j].col == temp[i].col ){
      if ( comp.elements[j].dist < temp[i].dist )
        elements.push_back(comp.elements[j]);
      else
        elements.push_back(temp[i]);
      i++;
      j++;
    }
    else {
      if ( component_id[temp[i].col][2] != id ) // No self edge
        elements.push_back(temp[i]);
      i++;
    }
  }

  // Remove any remaining edges to the new nodes
  while ( i < temp.size() ) {
    if ( temp[i].col == node ){
      weight += temp[i].dist;
    }
    else if ( component_id[temp[i].col][2] != id ) { // No self edge
      elements.push_back(temp[i]);
    }
    i++;
  }

  // Add any remaining elements at the end
  while ( j < comp.elements.size() ) {
    if ( component_id[comp.elements[j].col][2] != id ) { // No self edge
      elements.push_back(comp.elements[j]);
    }
    j++;
  }
}