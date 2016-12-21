#include "dopewild/cfhogwild.h"

#include "svm/svmmodel.h"
#include "svm/svm_exec.h"


using dopewild::CacheFriendlyHogwild;

using namespace hazy;
using namespace hazy::hogwild;
using namespace hazy::hogwild::svm;

typedef CacheFriendlyHogwild<SVMModel, SVMParams, SVMExec, SVMExample> SVMHogwild;

#include "svm_main.cc"
