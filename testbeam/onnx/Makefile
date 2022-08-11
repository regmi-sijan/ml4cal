all: onnxtestN onnxtest

onnxtest: onnxtest.C onnxsession.so onnxsession.h onnxutil.h
	g++ onnxtest.C ./onnxsession.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtest

onnxtestN: onnxtestN.C onnxsession.so onnxsession.h onnxutil.h
	g++ onnxtestN.C ./onnxsession.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtestN

onnxsession.so: onnxsession.C onnxsession.h
	g++ onnxsession.C -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -fPIC -shared -o onnxsession.so

clean:
	rm onnxtest onnxtestN onnxsession.so
