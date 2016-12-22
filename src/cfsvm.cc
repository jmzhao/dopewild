#include "dopewild/cfhogwild.h"

#include "svm/svmmodel.h"
#include "svm/svm_exec.h"


using dopewild::CacheFriendly;

using namespace hazy;
using namespace hazy::hogwild;
using namespace hazy::hogwild::svm;

typedef CacheFriendly<SVMModel, SVMParams, SVMExec, SVMExample>::CacheFriendlyHogwild SVMHogwild;

#include "svm_main.cc"
