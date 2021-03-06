#ifndef DOPEWILD_CFHOGWILD_H
#define DOPEWILD_CFHOGWILD_H

#include "hazy/vector/fvector.h"
#include "hazy/hogwild/hogwild-inl.h"
#include "hazy/hogwild/hogwild_task.h"

namespace dopewild {

using hazy::vector::FVector;
using hazy::hogwild::Hogwild;
using hazy::hogwild::HogwildTask;

/*! \brief Cache-friendly Hogwild! parallel executor
 */
template<class Model, class Params, class Exec, class Example>
class CacheFriendlyHogwild{

  struct CacheFriendlyModel {
    FVector<Model*> models;

    CacheFriendlyModel(Model &m, int count) {
      models.size = (unsigned) count;
      models.values = new Model*[count];
      models.values[0] = &m;
      for (int i = 1; i < count; i++) {
        models.values[i] = m.Clone();
      }
    }
  };

  class CacheFriendlyExec {
   public:
    static double UpdateModel(HogwildTask<CacheFriendlyModel, Params, Example> &cftask, unsigned tid, unsigned total) {
      HogwildTask<Model, Params, Example> task;
      task.model = cftask.model->models.values[tid];
      task.params = cftask.params;
      task.block = cftask.block;
      return Exec::UpdateModel(task, tid, total);
    }

    static double TestModel(HogwildTask<CacheFriendlyModel, Params, Example> &cftask, unsigned tid, unsigned total) {
      HogwildTask<Model, Params, Example> task;
      task.model = cftask.model->models.values[tid];
      task.params = cftask.params;
      task.block = cftask.block;
      return Exec::TestModel(task, tid, total);
    }

    static void PostUpdate(CacheFriendlyModel &cfmodel, Params &params) {
      //! Aggregate models into the first model
      Exec::Aggregate(cfmodel.models, params);

      Exec::PostUpdate(*cfmodel.models.values[0], params);

      for (int i = 1; i < cfmodel.models.size; i++) {
        cfmodel.models.values[i]->CopyFrom(*cfmodel.models.values[0]);
      }
    }

    static void PostEpoch(const CacheFriendlyModel &cfmodel, const Params &params) {
      for (int i = 0; i < cfmodel.models.size; i++) {
        Exec::PostEpoch(*cfmodel.models.values[i], params);
      }
    }
  };

 public:
  CacheFriendlyHogwild(Model &m, Params &p, hazy::thread::ThreadPool &tpool) :
      cfhogwild(*(new CacheFriendlyModel(m, tpool.ThreadCount())), p, tpool)
      {}


  template <class TrainScan, class TestScan>
  void RunExperiment(int nepochs, hazy::util::Clock &wall_clock,
                     TrainScan &trscan, TestScan &tescan) {
    cfhogwild.RunExperiment(nepochs, wall_clock, trscan, tescan);
  }

  template <class TrainScan>
  void RunExperiment(int nepochs, hazy::util::Clock &wall_clock,
                     TrainScan &trscan) {
    cfhogwild.RunExperiment(nepochs, wall_clock, trscan);
  }

 private:
  Hogwild<CacheFriendlyModel, Params, CacheFriendlyExec> cfhogwild;
};

} // namespace dopewild

#endif
