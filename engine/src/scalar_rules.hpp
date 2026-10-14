#pragma once

// expansion rules for the scalar zoom level, referenced from the op table in
// ops.cpp and defined in scalarize.cpp

#include "ag/scalarize.hpp"

namespace ag {

Status scalarizeElementwise(const ScalarizeArgs& a);
Status scalarizeMatMul(const ScalarizeArgs& a);
Status scalarizeReduce(const ScalarizeArgs& a);
Status scalarizeReshape(const ScalarizeArgs& a);
Status scalarizeDetach(const ScalarizeArgs& a);
Status scalarizeTranspose(const ScalarizeArgs& a);
Status scalarizeSoftmax(const ScalarizeArgs& a);
Status scalarizeMse(const ScalarizeArgs& a);
Status scalarizeCrossEntropy(const ScalarizeArgs& a);

} // namespace ag
