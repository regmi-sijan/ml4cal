all: onnxtestN

#onnxtest: onnxtest.C onnxlib.so onnxlib.h onnxutil.h
#	g++ onnxtest.C ./onnxlib.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtest

onnxtestN: onnxtestN.C onnxlib.so onnxlib.h onnxutil.h
	g++ onnxtestN.C ./onnxlib.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtestN

onnxlib.so: onnxlib.C onnxlib.h
	g++ onnxlib.C -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -fPIC -shared -o onnxlib.so

clean:
	rm onnxtest onnxtestN onnxlib.so
