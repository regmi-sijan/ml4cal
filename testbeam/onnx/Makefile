all: onnxsession.so onnxDriver onnxtestN onnxtest

onnxsession.so: onnxsession.C onnxsession.h
	g++ onnxsession.C -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -fPIC -shared -o onnxsession.so

onnxtest: onnxtest.C onnxsession.so onnxsession.h onnxutil.h
	g++ onnxtest.C ./onnxsession.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtest

onnxtestN: onnxtestN.C onnxsession.so onnxsession.h onnxutil.h
	g++ onnxtestN.C ./onnxsession.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxtestN

onnxDriver: onnxDriver.C onnxlib.so onnxlib.h onnxutil.h
	g++ onnxDriver.C ./onnxlib.so $(shell root-config --glibs --cflags --libs) -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -o onnxDriver

onnxlib.so: onnxlib.C onnxlib.h
	g++ onnxlib.C -L ./onnxruntime-linux-x64-1.11.1/lib -lonnxruntime -fPIC -shared -o onnxlib.so

clean:
	rm -f onnxtest onnxtestN onnxDriver onnxlib.so onnxsession.so
