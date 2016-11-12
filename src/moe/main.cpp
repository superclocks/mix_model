#include <gflags/gflags.h>
#include <iostream>
#include "config.h"
#include "moe.h"

using namespace std;
using namespace google;

DEFINE_string(input, "", "train file.");
DEFINE_string(model, "", "predict output file or model file.");
DEFINE_string(validation, "", "validate file.");
DEFINE_string(output, "", "predict file.");

DEFINE_double(l1, 0, "1-order bias L1 regularization");
DEFINE_double(l2, 0, "1-order bias L2 regularization");
DEFINE_double(alpha, 1.0, "sgd learning rate || ftrl 1-order bias learning rate");
DEFINE_double(beta, 0.2, "ftrl 1-order bias init accumulated gradient");

DEFINE_int32(k, 4, "number of sub-model.");
DEFINE_int32(n, 100000, "number of features");
DEFINE_int32(mode, 0, "train model or predict samples, 0 train, 1 predict");
DEFINE_int32(inter, 1, "train steps");
DEFINE_int32(normalize, 1, "normalize or not");
DEFINE_int32(debug, 0, "debug or not");
DEFINE_int32(thread_num, 2, "the number of openmp thread");

void parseArgs(Config& config)
{
  config._input = FLAGS_input;
  config._model = FLAGS_model;
  config._validation = FLAGS_validation;
  config._output = FLAGS_output;
  config._mode = FLAGS_mode;
  config._alfa = FLAGS_alpha;
  config._beta = FLAGS_beta;
  config._lamda1 = FLAGS_l1;
  config._lamda2 = FLAGS_l2;
  config._inter = FLAGS_inter;
  config._n = FLAGS_n;
  config._k = FLAGS_k;
  config._normalize = FLAGS_normalize;
  config._debug = FLAGS_debug;
  config._thread_num = FLAGS_thread_num;
}
int main(int argc, char** argv)
{
  ParseCommandLineFlags(&argc, &argv, true);  
  Config config;
  parseArgs(config);
  if(FLAGS_mode == 0)
  {
    //train model
    cout << "traing model." << endl;
    if(config._input.compare("") == 0 || config._model.compare("") == 0)
    {
      cerr<< "Pls set train file path and model file path." << endl;
    }
    MOE* moe = new MOE();
    moe->train(config);
    moe->saveModel();
  }
  else
  {
    //predict samples
    cout<< "predict samples." << endl;
    if(config._input.compare("") == 0 || config._model.compare("") == 0 || config._output.compare("") == 0)
    {
      cerr<< "Pls set predict file and model file and result file." << endl;
    }
    MOE* moe = new MOE();
    moe->predict(config);
  }
  return 0;
}
