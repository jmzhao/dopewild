# C++ Compiler and options
CPP=g++ -g -O3

# Path to dopewild (e.g. dopewildtl/include)
DOP_INCL=dopewildtl/include
# Path to hogwild (e.g. hogwildtl/include)
HOG_INCL=hogwildtl/include
# Path to Hazy Template Library (e.g. hazytl/include)
HTL_INCL=hazytl/include

# Conversion tools
TOOLS=bin/convert_matlab bin/convert bin/unconvert
UNAME=$(shell uname)
ifneq ($(UNAME), Darwin)
	LIB_RT=-lrt
endif

ALL= $(TOOLS) obj/frontend.o bin/svm bin/tracenorm bin/multicut \
		 bin/bbtracenorm bin/predict bin/bbsvm bin/bbmulticut \
		 bin/cfsvm

all: $(ALL)

obj/frontend.o:
	$(CPP) -c src/frontend_util.cc -o obj/frontend.o

bin/svm: obj/frontend.o
	$(CPP) -o bin/svm src/svm.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/bbsvm: obj/frontend.o
	$(CPP) -o bin/bbsvm src/bbsvm_main.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/tracenorm: obj/frontend.o
	$(CPP) -o bin/tracenorm src/tracenorm.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/predict:
	$(CPP) -o bin/predict src/tracenorm/predict.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/bbtracenorm: obj/frontend.o
	$(CPP) -o bin/bbtracenorm src/bbtracenorm.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/multicut: obj/frontend.o
	$(CPP) -o bin/multicut src/multicut.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/bbmulticut: obj/frontend.o
	$(CPP) -o bin/bbmulticut src/bbmulticut.cc -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

bin/convert: src/tools/tobinary.cc
	$(CPP) -o bin/convert src/tools/tobinary.cc -I$(HOG_INCL) -I$(HTL_INCL)

bin/convert_matlab: src/tools/tobinary.cc
	$(CPP) -o bin/convert_matlab src/tools/tobinary.cc -I$(HOG_INCL) -I$(HTL_INCL) -DMATLAB_CONVERT_OFFSET=1

bin/unconvert: src/tools/unconvert.cc
	$(CPP) -o bin/unconvert src/tools/unconvert.cc -I$(HOG_INCL) -I$(HTL_INCL)

bin/cfsvm: obj/frontend.o
	$(CPP) -o bin/cfsvm src/cfsvm.cc -I$(DOP_INCL) -I$(HOG_INCL) -I$(HTL_INCL) -lpthread $(LIB_RT) \
		obj/frontend.o

clean:
	rm -f $(ALL)
