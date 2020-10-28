//#include "imports.hpp"

PyObject* cgal_sdf(std::vector<K::Point_3> points, std::vector< std::vector<std::size_t> > polygons, int number_of_rays,double cone_angle)
{

    // create and read Polyhedron
    Polyhedron mesh;
    PyObject *my_sdf_list = PyList_New(0.0);

    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);

    if(mesh.empty()){
	PyList_Append(my_sdf_list, Py_BuildValue("d", 4.0));
        return my_sdf_list;
    }
    if(( !CGAL::is_triangle_mesh(mesh))){
        PyList_Append(my_sdf_list, Py_BuildValue("d", 6.0));
	return my_sdf_list;
    }

    // create a property-map for SDF values
    typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
    Facet_double_map internal_sdf_map;
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);

    // compute SDF values using default parameters for number of rays, and cone angle
    CGAL::sdf_values(mesh, sdf_property_map,cone_angle,number_of_rays);
    //CGAL::sdf_values(mesh, sdf_property_map);
    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
	facet_it != mesh.facets_end(); ++facet_it)
        PyList_Append(my_sdf_list, Py_BuildValue("d", sdf_property_map[facet_it]));

    return my_sdf_list;
}



static PyObject* cgal_SDF_C(PyObject *self, PyObject *args)
{
    int number_of_rays;
    double cone_angle;
    PyArrayObject*  v_array;
    PyArrayObject*  f_array;
    PyObject *my_sdf_list = PyList_New(0.0);

    if (!PyArg_ParseTuple(args,"O!O!id",&PyArray_Type,&v_array,&PyArray_Type,&f_array, &number_of_rays,&cone_angle))
    {
    	PyList_Append(my_sdf_list, Py_BuildValue("d", 111.0));
    	return my_sdf_list;
    }

    //READ VERTICES


    if (v_array->nd != 2 || v_array->descr->type_num != PyArray_DOUBLE) {
	PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type float");
	//return NULL;
        PyList_Append(my_sdf_list, Py_BuildValue("d", 222.0));
        return my_sdf_list;

    }

    double p,q,r;
    K::Point_3 mypoint;
    mypoint = K::Point_3(1,2,3);
    std::vector<K::Point_3> myvec;
    for (int i = 0; i < v_array->dimensions[0]; i++)
    {
	p = *(double *)(v_array->data + i*v_array->strides[0] + 0*v_array->strides[1]);
        q = *(double *)(v_array->data + i*v_array->strides[0] + 1*v_array->strides[1]);
        r = *(double *)(v_array->data + i*v_array->strides[0] + 2*v_array->strides[1]);
	mypoint = K::Point_3(p,q,r);
        myvec.push_back(mypoint);
    }

    //READ FACES

    if (f_array->nd != 2 || f_array->descr->type_num != PyArray_INT64) {
        PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type int");
        return NULL;
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

    //PyObject* ret_output_sdf = cgal_sdf(location_with_filename,number_of_clusters,smoothing_lambda);
    PyObject* ret_output_sdf = cgal_sdf(myvec, myfacevec,number_of_rays,cone_angle);

    return ret_output_sdf;

}
