
#include "sdf.hpp"
#include <Python.h>
#include "numpy/arrayobject.h"

#if 0
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <vector>
#endif

#if 1
#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;


//PyObject *cgal_segmentation(PyObject *self, PyObject *args)
//const char * cgal_segmentation(const char *location,const char * filename,int number_of_clusters,double smoothing_lambda)
//PyObject* cgal_sdf(const char* location_with_filename, int number_of_clusters,double smoothing_lambda)






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


PyObject* cgal_segmentation(std::vector<K::Point_3> points, std::vector< std::vector<std::size_t> > polygons, std::vector<double> sdfvector, int number_of_clusters,double smoothing_lambda)
{

    // create and read Polyhedron
    Polyhedron mesh;
    PyObject *my_list = PyList_New(0.0);

    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);

    if(mesh.empty()){
	PyList_Append(my_list, Py_BuildValue("d", 4.0));
        return my_list;
    }
    if(( !CGAL::is_triangle_mesh(mesh))){
        PyList_Append(my_list, Py_BuildValue("d", 6.0));
	return my_list;
    }



    // create a property-map for SDF values
    typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
    Facet_double_map internal_sdf_map;
    int sdfindex = 0;
    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin(); facet_it != mesh.facets_end(); ++facet_it)
    {
        //PyList_Append(my_list, Py_BuildValue("d", (double) segment_property_map[facet_it]));
        internal_sdf_map[facet_it] = sdfvector[sdfindex];
        sdfindex++;
    }
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);




    //std::vector<double> sdf_vec_values(mesh.size_of_facets());
    //Facet_with_id_pmap<double> sdf_property_map(sdf_vector);

    // compute SDF values using default parameters for number of rays, and cone angle
    //CGAL::sdf_values(mesh, sdf_property_map,cone_angle,number_of_rays);


    //COMPUTE SEGMENTATION
    // create a property-map for segment-ids
    typedef std::map<Polyhedron::Facet_const_handle, int> Facet_int_map;
    Facet_int_map internal_segment_map;


    boost::associative_property_map<Facet_int_map> segment_property_map(internal_segment_map);


    int number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);

     //WHAT TO RETURN

    for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
  facet_it != mesh.facets_end(); ++facet_it)
        PyList_Append(my_list, Py_BuildValue("d", (double) segment_property_map[facet_it]));

    return my_list;
}



static PyObject* cgal_segmentation_C(PyObject *self, PyObject *args)
{
    int number_of_clusters;
    double smoothing_lambda;
    PyArrayObject*  v_array;
    PyArrayObject*  f_array;
    PyArrayObject*  sdf_array;
    PyObject *my_list = PyList_New(0.0);

    if (!PyArg_ParseTuple(args,"O!O!O!id",&PyArray_Type,&v_array,&PyArray_Type,&f_array, &PyArray_Type,&sdf_array,&number_of_clusters,&smoothing_lambda))
    {
    	PyList_Append(my_list, Py_BuildValue("d", 111.0));
    	return my_list;
    }

    //READ VERTICES


    if (v_array->nd != 2 || v_array->descr->type_num != PyArray_DOUBLE) {
	PyErr_SetString(PyExc_ValueError,"array must be two-dimensional and of type float");
	//return NULL;
        PyList_Append(my_list, Py_BuildValue("d", 222.0));
        return my_list;

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
        //return NULL;
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


    //READ SDF

    if (sdf_array->nd != 1 || sdf_array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,"array must be one-dimensional and of type double");
        PyList_Append(my_list, Py_BuildValue("d", 678.0));
        return my_list;
    }

    double sdfval;
    std::vector<double > mysdfvec;
    mysdfvec.clear();
    for (int i = 0; i < sdf_array->dimensions[0]; i++)

    {
	    sdfval = *(double *)(sdf_array->data + i*sdf_array->strides[0] );//+ j*f_array->strides[1]);
      mysdfvec.push_back(sdfval);
    }





    //PyObject* ret_output_sdf = cgal_sdf(location_with_filename,number_of_clusters,smoothing_lambda);
    PyObject* ret_output = cgal_segmentation(myvec, myfacevec,mysdfvec,number_of_clusters,smoothing_lambda);
    //PyObject* ret_output = cgal_sdf(myvec, myfacevec,5,2.0);
    return ret_output;

}



static PyMethodDef cgal_SDF_Methods[] =
{
    { "cgal_sdf", cgal_SDF_C, METH_VARARGS, "calculates the sdf" },
    {"cgal_segmentation", cgal_segmentation_C, METH_VARARGS, "calculates the segmentation"},
    { NULL,NULL,0, NULL }

};




static struct PyModuleDef cgal_SDF_Module =
{
    PyModuleDef_HEAD_INIT,
    "cgal_Sdf_Module",
    "CGAL SDF Module",
    -1,
    cgal_SDF_Methods
};

PyMODINIT_FUNC PyInit_cgal_SDF_Module(void )
{
    import_array();
    return PyModule_Create(&cgal_SDF_Module);
}


/*PyMODINIT_FUNC
PyInit_numpytest(void)
{
    import_array();
    return PyModule_Create(&numpytest);
}*/
