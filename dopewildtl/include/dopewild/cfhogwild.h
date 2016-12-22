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
struct CacheFriendly{

  struct CacheFriendlyModel {
    FVector<Model*> models;
    hazy::util::Clock *p_aggregate_time_; //!< measures the time spent in most recent Aggregate
    hazy::util::Clock *p_train_time_; //!< measures the time spent in most recent Aggregate

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

    static void PostEpoch(const CacheFriendlyModel &cfmodel, const Params &params) {
      for (int i = 0; i < cfmodel.models.size; i++) {
        Exec::PostEpoch(*cfmodel.models.values[i], params);
      }
    }
  };

  class CacheFriendlyHogwild : public Hogwild<CacheFriendlyModel, Params, CacheFriendlyExec> {
   public:
    CacheFriendlyHogwild(Model &m, Params &p, hazy::thread::ThreadPool &tpool) :
        Hogwild(*(new CacheFriendlyModel(m, tpool.ThreadCount())), p, tpool)
        {}

   public:
    void PostUpdate(CacheFriendlyModel &cfmodel, Params &params) {
      //! Aggregate models into the first model
      hogwild_.train_time_.Start();
      aggregate_time_.Start();
      Exec::Aggregate(cfmodel.models, params);
      aggregate_time_.Pause();
      hogwild_.train_time_.Pause();

      Exec::PostUpdate(*cfmodel.models.values[0], params);

      hogwild_.train_time_.Start();
      aggregate_time_.Start();
      for (int i = 1; i < cfmodel.models.size; i++) {
        cfmodel.models.values[i]->CopyFrom(*cfmodel.models.values[0]);
      }
      aggregate_time_.Stop();
      hogwild_.train_time_.Pause();
    }

    template <class TrainScan, class TestScan>
    void RunExperiment(
        int nepochs, hazy::util::Clock &wall_clock,
        TrainScan &trscan, TestScan &tescan) {
      printf("wall_clock: %.5f    Going CFHogwild!\n", wall_clock.Read());
      for (int e = 1; e <= nepochs; e++) {
        UpdateModel(trscan);
        PostUpdate(model_, params_);
        double train_rmse = ComputeRMSE(trscan);
        double test_rmse = ComputeRMSE(tescan);

        printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5f aggregate_time: %.5f train_rmse: %.5f test_rmse: %.5f\n",
               e, wall_clock.Read(), train_time_.value, test_time_.value,
               epoch_time_.value, aggregate_time_.value, train_rmse, test_rmse);
        fflush(stdout);

        CacheFriendlyExec::PostEpoch(model_, params_);
      }
    }

    template <class TrainScan>
    void RunExperiment(
        int nepochs, hazy::util::Clock &wall_clock, TrainScan &trscan) {
      printf("wall_clock: %.5f    Going CFHogwild!\n", wall_clock.Read());
      for (int e = 1; e <= nepochs; e++) {
        UpdateModel(trscan);
        PostUpdate(model_, params_);
        double train_rmse = ComputeRMSE(trscan);

        printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5f aggregate_time: %.5f train_rmse: %.5f\n",
               e, wall_clock.Read(), train_time_.value, test_time_.value,
               epoch_time_.value, aggregate_time_.value, train_rmse);
        fflush(stdout);

        CacheFriendlyExec::PostEpoch(model_, params_);
      }
    }

    // template <class TrainScan, class TestScan>
    // void RunExperiment(int nepochs, hazy::util::Clock &wall_clock,
    //                    TrainScan &trscan, TestScan &tescan) {
    //   hogwild_.RunExperiment(nepochs, wall_clock, trscan, tescan);
    // }
    //
    // template <class TrainScan>
    // void RunExperiment(int nepochs, hazy::util::Clock &wall_clock,
    //                    TrainScan &trscan) {
    //   hogwild_.RunExperiment(nepochs, wall_clock, trscan);
    // }

   private:
    Hogwild<CacheFriendlyModel, Params, CacheFriendlyExec> hogwild_;
    hazy::util::Clock aggregate_time_;
  };
};

} // namespace dopewild

#endif
