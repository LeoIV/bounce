#!/usr/bin/bash

for BENCHMARK in "@SVMMixed()" "@Contamination()" "@Labs()" "@MaxSat60()" "@Ackley53()" "@PestControl()"; do
  for FLIP in True False; do
    for BS in 1 2 3 5 10; do
      singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
        --gin-bindings \"Bounce.benchmark = $BENCHMARK\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = $BS\' \
        \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 250\' \
        \'Bounce.initial_target_dimensionality = 5\'
    done
  done
done

for BENCHMARK in "@SVMMixed()" "@Contamination()" "@Labs()" "@MaxSat60()" "@Ackley53()" "@PestControl()"; do
  for FLIP in True False; do
    singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
      --gin-bindings \"Bounce.benchmark = $BENCHMARK\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = 20\' \
      \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 500\' \
      \'Bounce.initial_target_dimensionality = 5\'
  done
done

for BENCHMARK in "@SVM()" "@Mopta08()" "@HartmannEffectiveDim()" "@BraninEffectiveDim()" "@LassoDNA()" "@LassoHard() @LassoHigh()"; do
  for FLIP in True False; do
    for BS in 1 2 3 5 10; do
      singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
        --gin-bindings \"Bounce.benchmark = $BENCHMARK\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = $BS\' \
        \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 250\' \
        \'Bounce.initial_target_dimensionality = 2\'
    done
  done
done

for BENCHMARK in "@SVM()" "@Mopta08()" "@HartmannEffectiveDim()" "@BraninEffectiveDim()" "@LassoDNA()" "@LassoHard()" "@LassoHigh()"; do
  for FLIP in True False; do
    singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
      --gin-bindings \"Bounce.benchmark = $BENCHMARK\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = 20\' \
      \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 500\' \
      \'Bounce.initial_target_dimensionality = 2\'
  done
done

for FLIP in True False; do
  for BS in 1 2 3 5 10; do
    singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
      --gin-bindings \"Bounce.benchmark = "@MaxSat125()"\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = $BS\' \
      \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 250\' \
      \'Bounce.initial_target_dimensionality = 5\' \'MaxSat.normalize_weights = False\'
  done
done

for FLIP in True False; do
  singularity run --nv --bind results:/bs/bounce/results bounce.sif --gin-files configs/default.gin \
    --gin-bindings \"Bounce.benchmark = "@MaxSat125()"\" \'Benchmark.flip = $FLIP\' \'Bounce.batch_size = 20\' \
    \'Bounce.maximum_number_evaluations = 2000\' \'Bounce.maximum_number_evaluations_until_input_dim = 500\' \
    \'Bounce.initial_target_dimensionality = 5\' \'MaxSat.normalize_weights = False\'
done
