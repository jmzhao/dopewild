// Copyright 2012 Victor Bittorf, Chris Re
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Hogwild!, part of the Hazy Project
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)
// Original Hogwild! Author: Chris Re (chrisre [at] cs.wisc.edu)             

#ifndef HAZY_HOGWILD_INSTANCES_SVM_MODEL_H
#define HAZY_HOGWILD_INSTANCES_SVM_MODEL_H

#include "hazy/vector/fvector.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/operations-inl.h"

#include "hazy/hogwild/hogwild_task.h"

namespace hazy {
namespace hogwild {
//! Sparse SVM implementation
namespace svm {


//! The precision of the values, either float or double
typedef double fp_type;

//! The mutable model for a sparse SVM.
struct SVMModel {
  //! The weight vector that is trained
  vector::FVector<fp_type> weights;

  //! Construct a weight vector of length dim backed by the buffer
  /*! A new model backed by the buffer.
   * \param buf the backing memory for the weight vector
   * \param dim the length of buf
   */
  explicit SVMModel(unsigned dim) {
    weights.values = new fp_type[dim];
    weights.size = dim;
    for (unsigned i = dim; i-- > 0; ) {
      weights.values[i] = 0;
    }
  }

  /*! Copies from the given model into this model.
   * This is required for the 'Best Ball' training.
   */
  void CopyFrom(SVMModel const &m) {
    assert(weights.size == m.weights.size);
    vector::CopyInto(m.weights, weights);
  }

  /*! Creates a deep copy of this model, caller must free.
   * This is required for the 'Best Ball' training.
   */
  SVMModel* Clone() {
    SVMModel *m = new SVMModel(weights.size);
    m->CopyFrom(*this);
    return m;
  }
};

//! Parameters for SVM training
struct SVMParams {
  float mu; //!< mu param
  float step_size; //!< stepsize (decayed by step decay at each epoch)
  float step_decay; //!< factor to modify step_size by each epoch
  unsigned const *degrees; //!< degree of each feature
  unsigned ndim; //!< number of features, length of degrees

  //! Constructs a enw set of params
  SVMParams(fp_type stepsize, fp_type stepdecay, fp_type _mu) :
      mu(_mu), step_size(stepsize), step_decay(stepdecay) { }
};

//! A single example which is a value/rating and a vector
struct SVMExample {
  fp_type value; //!< rating of this example
  vector::SVector<const fp_type> vector; //!< feature vector

  SVMExample() { }
  //! Constructs a new example
  /*! Makes an example backed by the given memory.
   * \param val example's rating
   * \param values the values of the features (for a sprase vector)
   * \param index the indicies of the values
   * \param len the length of index and values
   */
  SVMExample(fp_type val, fp_type const * values, int * index, 
             unsigned len) :
      value(val), vector(values, index, len) { }

  SVMExample(const SVMExample &o) {
    value = o.value;
    vector.values = o.vector.values;
    vector.index = o.vector.index;
    vector.size = o.vector.size;
  }
};

typedef HogwildTask<SVMModel, SVMParams, SVMExample> SVMTask;

} // namespace svm
} // namespace hogwild

} // namespace hazy

#endif
