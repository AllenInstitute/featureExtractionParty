
struct Stats
{
  Stats()
    : collected(0)
    , processed(0)
    , collapsed(0)
    , non_collapsable(0)
    , cost_uncomputable(0)
    , placement_uncomputable(0)
  {}

  std::size_t collected ;
  std::size_t processed ;
  std::size_t collapsed ;
  std::size_t non_collapsable ;
  std::size_t cost_uncomputable  ;
  std::size_t placement_uncomputable ;
} ;

struct My_visitor : SMS::Edge_collapse_visitor_base<Surface_mesh>
{
  My_visitor( Stats* s) : stats(s){}

  // Called during the collecting phase for each edge collected.
  void OnCollected( Profile const&, boost::optional<double> const& )
  {
    ++ stats->collected ;
    std::cerr << "\rSharmi test Edges collected: " << stats->collected << std::flush ;
  }


  // Called during the processing phase for each edge selected.
  // If cost is absent the edge won't be collapsed.
  void OnSelected(Profile const&
                 ,boost::optional<double> cost
                 ,std::size_t             initial
                 ,std::size_t             current
                 )
  {
    ++ stats->processed ;
    if ( !cost )
      ++ stats->cost_uncomputable ;

    if ( current == initial )
      std::cerr << "\n" << std::flush ;
    std::cerr << "\r" << current << std::flush ;
  }

  // Called during the processing phase for each edge being collapsed.
  // If placement is absent the edge is left uncollapsed.
  void OnCollapsing(Profile const&
                   ,boost::optional<Point>  placement
                   )
  {
    if ( !placement )
      ++ stats->placement_uncomputable ;
    std::cerr<<"    Placement uncomputable: ";
    std::cerr<<stats->placement_uncomputable<<std::flush;
  }

  // Called for each edge which failed the so called link-condition,
  // that is, which cannot be collapsed because doing so would
  // turn the surface mesh into a non-manifold.
  void OnNonCollapsable( Profile const& )
  {
    ++ stats->non_collapsable;
    std::cerr<<"Non Collapsable: ";
    std::cerr<<stats->non_collapsable<<std::flush;
  }

  // Called AFTER each edge has been collapsed
  void OnCollapsed( Profile const&, vertex_descriptor )
  {
    ++ stats->collapsed;
    std::cerr<<"     collapsed: ";
    std::cerr<<stats->collapsed<<std::flush;
  }

  Stats* stats ;
} ;


PyObject* cgal_simplify(std::vector<Point_3> &points, std::vector< std::vector<std::size_t> > &polygons, double scale)

{
  PyObject *my_vertices = PyList_New(0.0);
  PyObject *my_faces = PyList_New(0.0);
  PyObject *my_object = PyList_New(0.0);

  // Read surface mesh
  Surface_mesh sm;

  for (std::size_t i = 0, end = points.size(); i < end; ++i)
    {
      const Point_3 & p = points[i];
      sm.add_vertex(p);
    }
  for (std::size_t i = 0, end = polygons.size(); i < end; ++i)
      sm.add_face(polygons[i]);

  //keep full res mesh in memory and perform simplification
  Surface_mesh surface_meshfull(sm);
  SMS::Count_ratio_stop_predicate<Surface_mesh> stop(scale);
  Stats stats ;
  My_visitor vis(&stats) ;

  int r = SMS::edge_collapse
           (sm
           ,stop
            ,CGAL::parameters::get_cost     (SMS::Edge_length_cost  <Surface_mesh>())
                              .get_placement(SMS::Midpoint_placement<Surface_mesh>())
                              .visitor      (vis)
           );

  //output
  std::vector<int> reindex;
  int n = 0;
  reindex.resize(sm.num_vertices());

  Surface_mesh::Vertex_range V = sm.vertices();

  for (Vertex_iterator it = V.begin() ; it != V.end(); ++it)
  {
    Point_3 mypoint = sm.point(*it);
    PyObject *my_pt = PyList_New(0.0);
    PyList_Append(my_pt, Py_BuildValue("d",   mypoint.x()));
    PyList_Append(my_pt, Py_BuildValue("d",   mypoint.y()));
    PyList_Append(my_pt, Py_BuildValue("d",   mypoint.z()));
    PyList_Append(my_vertices, Py_BuildValue("O",   my_pt));
    reindex[*it]=n++;
  }

  for(Face_index f : sm.faces())
  {
      PyObject *my_face = PyList_New(0.0);
      for(Vertex_index v : CGAL::vertices_around_face(sm.halfedge(f),sm))
      {
        PyList_Append(my_face, Py_BuildValue("i",   reindex[v] ));
      }
      PyList_Append(my_faces, Py_BuildValue("O",   my_face));
  }


  PyList_Append(my_object, Py_BuildValue("O",   my_vertices));
  PyList_Append(my_object, Py_BuildValue("O",   my_faces));
  return my_object;

}


static PyObject* cgal_simplify_C(PyObject *self, PyObject *args)
{

  double scale;
  PyArrayObject*  v_array;
  PyArrayObject*  f_array;

  PyObject *my_list = PyList_New(0.0);

  if (!PyArg_ParseTuple(args,"O!O!d",&PyArray_Type,&v_array,&PyArray_Type,&f_array, &scale))
  {
    PyList_Append(my_list, Py_BuildValue("d", 0.0));
    return my_list;
  }

    //READ VERTICES
      if (v_array->nd != 2 || v_array->descr->type_num != PyArray_DOUBLE)
      {
          PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type float");
          PyList_Append(my_list, Py_BuildValue("d", 222.0));
          return my_list;

      }

      double p,q,r;
      SCKernel::Point_3 mypoint;
      mypoint = SCKernel::Point_3(1,2,3);
      std::vector<SCKernel::Point_3> myvec;
      for (int i = 0; i < v_array->dimensions[0]; i++)
      {
          p = *(double *)(v_array->data + i*v_array->strides[0] + 0*v_array->strides[1]);
          q = *(double *)(v_array->data + i*v_array->strides[0] + 1*v_array->strides[1]);
          r = *(double *)(v_array->data + i*v_array->strides[0] + 2*v_array->strides[1]);
          mypoint = SCKernel::Point_3(p,q,r);
          myvec.push_back(mypoint);
      }

      //READ FACES

      if (f_array->nd != 2 || f_array->descr->type_num != PyArray_INT64)
      {
          PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type int");
          PyList_Append(my_list, Py_BuildValue("d", 345.0));
          return my_list;
      }

      int u;
      std::vector<std::size_t>  face;
      std::vector<std::vector<std::size_t> > myfacevec;

      for (int i = 0; i < f_array->dimensions[0]; i++)
      {
          face.clear();
          for (int j = 0; j < f_array->dimensions[1]; j++)
          {
              u = *(std::size_t *)(f_array->data + i*f_array->strides[0] + j*f_array->strides[1]);
              face.push_back(u);
          }
          myfacevec.push_back(face);
      }

      //return ret_output;
      PyObject* ret_output = cgal_simplify(myvec,myfacevec,scale);
      return ret_output;

}
