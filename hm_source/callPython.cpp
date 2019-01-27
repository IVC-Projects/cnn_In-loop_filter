#include "stdio.h"
#include <Python.h>
#include <windows.h>
#include <direct.h>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

extern void conv_y_to_mat(cv::Mat &YYY, unsigned char* pY, int nWidth, int nHeight, int bit_depth);

cv::Mat callPython(cv::Mat img ,int ccc_type){


	int height = img.rows;
	int width = img.cols;
	cv::Mat recMat;

	PyObject * pModule = NULL;
	PyObject * pFuncI = NULL;
	PyObject * pFuncP = NULL;
	PyObject * pFuncB = NULL;
	PyObject * pArgs = NULL;

	Py_Initialize();


	pModule = PyImport_ImportModule("TEST_qp37");

	if (!pModule) {
		printf("don't get Pmodule\n");
                PyErr_Print();
		Py_Finalize();
		return recMat;
	}
	pFuncI = PyObject_GetAttrString(pModule, "modelI");
	if (!pFuncI) {
		printf("don't get I function!");
		Py_Finalize();
		return recMat;
	}
	pFuncP = PyObject_GetAttrString(pModule, "modelP");
	if (!pFuncP) {
		printf("don't get P function!");
		Py_Finalize();
		return recMat;
	}

	pFuncB = PyObject_GetAttrString(pModule, "modelB");

	if (!pFuncB) {
		printf("don't get B function!");
		Py_Finalize();
		return recMat;
	}
	
	PyObject* list = PyList_New(height); 
	pArgs = PyTuple_New(1);                 
	PyObject** lists = new PyObject*[height];
	uchar *temp;
	for (int i = 0; i < height; i++)
	{
		temp = img.ptr<uchar>(i);
		lists[i] = PyList_New(0);
		for (int j = 0; j < width; j++)
		{
			PyList_Append(lists[i], Py_BuildValue("i", temp[j]));
		}
		PyList_SetItem(list, i, lists[i]);
	}
	PyTuple_SetItem(pArgs, 0, list);    
	
        PyObject *presult = NULL;
        if (ccc_type == 1){
	    printf("c++, in I\n");
	    presult = PyEval_CallObject(pFuncI, pArgs);
        }
        else if (ccc_type == 2){
	    printf("c++, in P\n");
	    presult = PyEval_CallObject(pFuncP, pArgs);
        }
        else{
		printf("c++, in B\n");
	    presult = PyEval_CallObject(pFuncB, pArgs);
		}
	Py_ssize_t high = PyList_Size(presult), wid = PyList_Size(PyList_GetItem(presult, 0));
	int buflen = high * wid;
	unsigned char *buf = new unsigned char[buflen];
	unsigned char *p = buf;
	int s;

	for (int i = 0; i < high; i++)
	{
		for (int j = 0; j < wid; j++)
		{
			PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "i", &s); 
			*p = s;
			p++;
		}
	}

	
	conv_y_to_mat(recMat, buf, wid, high, 8);
	return recMat;
}
