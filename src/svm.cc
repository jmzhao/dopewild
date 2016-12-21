#include "hazy/hogwild/hogwild-inl.h"

#include "svm/svmmodel.h"
#include "svm/svm_loader.h"
#include "svm/svm_exec.h"

using namespace hazy;
using namespace hazy::hogwild;
using namespace hazy::hogwild::svm;

typedef Hogwild<SVMModel, SVMParams, SVMExec> SVMHogwild;

#include "svm_main.cc"
