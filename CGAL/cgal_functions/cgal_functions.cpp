

#include "imports.hpp"
#include "cgal_functions.hpp"
#include "segmentation.hpp"
#include "sdf.hpp"
#include "simplify.hpp"
#include "low2highseg.hpp"
/*#include <Python.h>
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



typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;*/


//PyObject *cgal_segmentation(PyObject *self, PyObject *args)
//const char * cgal_segmentation(const char *location,const char * filename,int number_of_clusters,double smoothing_lambda)
//PyObject* cgal_sdf(const char* location_with_filename, int number_of_clusters,double smoothing_lambda)









static PyMethodDef cgal_functions_Methods[] =
{
    { "cgal_sdf", cgal_SDF_C, METH_VARARGS, "calculates the sdf" },
    {"cgal_segmentation", cgal_segmentation_C, METH_VARARGS, "calculates the segmentation"},
    {"cgal_simplify", cgal_simplify_C, METH_VARARGS, "simplifies mesh"},
    {"cgal_low2highseg", cgal_low2highseg_C, METH_VARARGS, "maps low res segmentation to high res"},
    { NULL,NULL,0, NULL }

};




static struct PyModuleDef cgal_functions_Module =
{
    PyModuleDef_HEAD_INIT,
    "cgal_functions_Module",
    "CGAL Functions Module",
    -1,
    cgal_functions_Methods
};

PyMODINIT_FUNC PyInit_cgal_functions_Module(void )
{
    import_array();
    return PyModule_Create(&cgal_functions_Module);
}


/*PyMODINIT_FUNC
PyInit_numpytest(void)
{
    import_array();
    return PyModule_Create(&numpytest);
}*/
