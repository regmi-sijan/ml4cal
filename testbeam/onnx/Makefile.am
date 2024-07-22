AUTOMAKE_OPTIONS = foreign

AM_CPPFLAGS = \
  -I$(includedir) \
  -isystem$(OFFLINE_MAIN)/include \
  -isystem$(ROOTSYS)/include

AM_LDFLAGS = \
  -L$(libdir) \
  -L$(OFFLINE_MAIN)/lib \
  -L$(OFFLINE_MAIN)/lib64 \
  `root-config --glibs --cflags --libs`

bin_PROGRAMS = \
  onnxDriver \
  onnxtestN \
  onnxtest

onnxDriver_SOURCES = onnxDriver.C
onnxtestN_SOURCES = onnxtestN.C
onnxtest_SOURCES = onnxtest.C

onnxDriver_LDADD = libonnx.la
onnxtestN_LDADD = libonnx.la
onnxtest_LDADD = libonnx.la

lib_LTLIBRARIES = \
  libonnx.la

libonnx_la_SOURCES = \
  onnxlib.C \
  onnxsession.C

libonnx_la_LIBADD = \
  -lonnxruntime
