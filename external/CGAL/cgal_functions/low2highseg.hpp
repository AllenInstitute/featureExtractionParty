std::vector<SCKernel::Point_3> read_vertices(PyArrayObject* v_array)
{
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

        return myvec;
}

std::vector<std::vector<std::size_t> > read_faces(PyArrayObject * f_array)
{
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
      return myfacevec;
}

std::vector<int> read_segmentation (PyArrayObject *seg_array)
{
      double segval;
      std::vector<int> mysegvec;
      mysegvec.clear();
      for (int i = 0; i < seg_array->dimensions[0]; i++)

      {
        segval = *(int *)(seg_array->data + i*seg_array->strides[0] );//+ j*f_array->strides[1]);
        mysegvec.push_back(segval);
      }

      return mysegvec;
}

Surface_mesh get_surface_mesh(std::vector<Point_3> points, std::vector< std::vector<std::size_t> > polygons)
{
      Surface_mesh sm;

      for (std::size_t i = 0, end = points.size(); i < end; ++i)
        {
          const Point_3 & p = points[i];
          sm.add_vertex(p);
        }
      for (std::size_t i = 0, end = polygons.size(); i < end; ++i)
          sm.add_face(polygons[i]);




      return sm;

}



PyObject* cgal_low2highseg(std::vector<Point_3> points, std::vector< std::vector<std::size_t> > polygons, std::vector<Point_3> points_low, std::vector< std::vector<std::size_t> > polygons_low, std::vector<int> seg_low)

{
      //input
      PyObject *my_object = PyList_New(0.0);
      //std::vector<int> seg_full;

      //read high res mesh_segmentation
      Surface_mesh surface_meshfull = get_surface_mesh (points, polygons);

      //read low res mesh_segmentation
      Surface_mesh surface_mesh = get_surface_mesh (points_low, polygons_low);

      // read segmentation into property map

      typedef Surface_mesh::Property_map<face_descriptor, std::size_t> Facet_int_map;
      Facet_int_map segment_property_map = surface_mesh.add_property_map<face_descriptor,std::size_t>("f:sid").first;;

      std::vector<int>::iterator label_it = seg_low.begin();
      Face_iterator facet_it, fend;
      for(boost::tie(facet_it,fend) = faces(surface_mesh); facet_it != fend; ++facet_it, ++label_it)
      {
            put(segment_property_map, *facet_it, *label_it);
      }

      //assign with Tree

      Tree tree(faces(surface_mesh).first, faces(surface_mesh).second, surface_mesh);
      tree.accelerate_distance_queries();

      Vertex_iterator vb,ve;

      for(boost::tie(vb, ve) = surface_meshfull.vertices(); vb!=ve; ++vb){
          Point_and_primitive_id pp = tree.closest_point_and_primitive(surface_meshfull.point(*vb));
          PyList_Append(my_object, Py_BuildValue("i", segment_property_map[pp.second]));
      }

      return my_object;

}


static PyObject* cgal_low2highseg_C(PyObject *self, PyObject *args)
{
      PyArrayObject*  v_array;
      PyArrayObject*  f_array;
      PyArrayObject*  vlow_array;
      PyArrayObject*  flow_array;
      PyArrayObject*  seg_array;

      PyObject *my_list = PyList_New(0.0);

      if (!PyArg_ParseTuple(args,"O!O!O!O!O!",&PyArray_Type,&v_array,&PyArray_Type,&f_array, &PyArray_Type,&vlow_array,&PyArray_Type,&flow_array,&PyArray_Type,&seg_array))
      {
        PyList_Append(my_list, Py_BuildValue("d", 0.0));
        return my_list;
      }

      //CHECK
      if (v_array->nd != 2 || v_array->descr->type_num != PyArray_DOUBLE || vlow_array->nd != 2 || vlow_array->descr->type_num != PyArray_DOUBLE)
      {
          PyErr_SetString(PyExc_ValueError," Vertex array must be two-dimensional and of type float");
          PyList_Append(my_list, Py_BuildValue("d", 222.0));
          return my_list;

      }

      if (f_array->nd != 2 || f_array->descr->type_num != PyArray_INT64 || flow_array->nd != 2 || flow_array->descr->type_num != PyArray_INT64)
      {
          PyErr_SetString(PyExc_ValueError,"Face array must be two-dimensional and of type int");
          PyList_Append(my_list, Py_BuildValue("d", 345.0));
          return my_list;
      }

      if (seg_array->nd != 1 || seg_array->descr->type_num != PyArray_INT64) {
          PyErr_SetString(PyExc_ValueError,"Seg array must be one-dimensional and of type int");
          PyList_Append(my_list, Py_BuildValue("d", 678.0));
          return my_list;
      }

      //READ ALL OBJECTS
      std::vector<SCKernel::Point_3> myvec = read_vertices(v_array);
      std::vector<SCKernel::Point_3> myveclow = read_vertices(vlow_array);
      std::vector<std::vector<std::size_t> > myfacevec = read_faces(f_array);
      std::vector<std::vector<std::size_t> > myfaceveclow = read_faces(flow_array);
      std::vector<int> mysegvec = read_segmentation(seg_array);

      //return ret_output;
      PyObject* ret_output = cgal_low2highseg(myvec,myfacevec,myveclow, myfaceveclow, mysegvec);
      return ret_output;

}
