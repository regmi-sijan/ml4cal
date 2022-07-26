onnxtest: onnxtest.C onnxlib.so onnxlib.h onnxutil.h
	g++ onnxtest.C ./onnxlib.so $(shell root-config --glibs --cflags --libs) -L${LD_LIBRARY_PATH} -lonnxruntime -o onnxtest

onnxlib.so: onnxlib.C onnxlib.h
	g++ onnxlib.C -L${LD_LIBRARY_PATH} -lonnxruntime -fPIC -shared -o onnxlib.so
