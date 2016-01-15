#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "nn_like.h"


static PyObject *nn_like_nn_like(PyObject *self, PyObject *args);
static PyObject *nn_like_forward_deterministic(PyObject *self, PyObject *args);
static PyObject *nn_like_forward(PyObject *self, PyObject *args);
static PyObject *nn_like_backprop_deterministic(PyObject *self, PyObject *args);
static PyObject *nn_like_print_states(PyObject *self, PyObject *args);
static PyObject *nn_like_print_connections(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"nn_like", nn_like_nn_like, METH_VARARGS, "Creates a NN."},
    {"forward_deterministic", nn_like_forward_deterministic, METH_VARARGS, "Applies the NN."},
    {"forward", nn_like_forward, METH_VARARGS, "Applies the NN."},
    {"backprop_deterministic", nn_like_backprop_deterministic, METH_VARARGS, "Backprop deterministic."},
    {"print_states", nn_like_print_states, METH_VARARGS, "Print states."},
    {"print_connections", nn_like_print_connections, METH_VARARGS, "Print connections."},
    {NULL, NULL, 0, NULL}
};

// initialize module
PyMODINIT_FUNC PyInit__nn_like(void) {
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_nn_like",          /* m_name */
        "A C implementation for a NN with weight uncertainties.",  /* m_doc */
        -1,                  /* m_size */
        module_methods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    import_array();
    return m;
}

static PyObject *nn_like_nn_like(PyObject *self, PyObject *args) {
    PyObject *layer_units_obj;

    // parse input tuple
    if (!PyArg_ParseTuple(args, "O", &layer_units_obj))
        return NULL;

    // input to numpy arrays
    PyArrayObject *layer_units_array = (PyArrayObject*)PyArray_FROM_OTF(layer_units_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (layer_units_array == NULL) {
        Py_XDECREF(layer_units_array);
        return NULL;
    }

    // input to C data
    int n_layers = (int)PyArray_DIMS(layer_units_array)[0];
    double *layer_units_d = (double*)PyArray_DATA(layer_units_array);
    int layer_units[n_layers];
    for (int i=0; i < n_layers; i++) layer_units[i] = (int)layer_units_d[i];

    // C call
    nn_like(n_layers, layer_units);

    Py_DECREF(layer_units);
    PyObject *ret = Py_BuildValue("d", 0.0);
    return ret;
}

static PyObject *nn_like_forward_deterministic(PyObject *self, PyObject *args) {
    PyObject *input_obj;

    // parse input tuple
    if (!PyArg_ParseTuple(args, "O", &input_obj))
        return NULL;

    // to numpy array
    PyArrayObject *input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        Py_XDECREF(input_array);
        return NULL;
    }

    // allocate output memory
    int dims[1] = {output_size()};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    double* outputs = (double*)PyArray_DATA(output_array);

    // input to C data
    double *inputs = (double*)PyArray_DATA(input_array);

    // C call
    forward_deterministic(inputs, outputs);

    Py_DECREF(inputs);
    PyObject *ret = Py_BuildValue("O", output_array);
    return ret;
}

static PyObject *nn_like_forward(PyObject *self, PyObject *args) {
    PyObject *input_obj;

    // parse input tuple
    if (!PyArg_ParseTuple(args, "O", &input_obj))
        return NULL;

    // to numpy array
    PyArrayObject *input_array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (input_array == NULL) {
        Py_XDECREF(input_array);
        return NULL;
    }

    // allocate output memory
    int dims[1] = {output_size()};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    double* outputs = (double*)PyArray_DATA(output_array);

    // input to C data
    double *inputs = (double*)PyArray_DATA(input_array);

    // C call
    forward(inputs, outputs);

    Py_DECREF(inputs);
    PyObject *ret = Py_BuildValue("O", output_array);
    return ret;
}

static PyObject *nn_like_backprop_deterministic(PyObject *self, PyObject *args) {
    PyObject *outputs_obj;
    PyObject *targets_obj;
    double eta;

    // parse input tuple
    if (!PyArg_ParseTuple(args, "OOd", &outputs_obj, &targets_obj, &eta))
        return NULL;

    // to numpy array
    PyArrayObject *output_array = (PyArrayObject*)PyArray_FROM_OTF(outputs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *target_array = (PyArrayObject*)PyArray_FROM_OTF(targets_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (output_array == NULL || target_array == NULL) {
        Py_XDECREF(output_array);
        Py_XDECREF(target_array);
        return NULL;
    }

    // input to C data
    double *outputs = (double*)PyArray_DATA(output_array);
    double *targets = (double*)PyArray_DATA(target_array);

    // C call
    backprop_deterministic(outputs, targets, eta);

    Py_DECREF(outputs);
    Py_DECREF(targets);
    PyObject *ret = Py_BuildValue("O", output_array);
    return ret;
}

static PyObject *nn_like_print_states(PyObject *self, PyObject *args) {
    print_states();

    PyObject *ret = Py_BuildValue("d", 0.0);
    return ret;
}

static PyObject *nn_like_print_connections(PyObject *self, PyObject *args) {
    print_connections();

    PyObject *ret = Py_BuildValue("d", 0.0);
    return ret;
}
