//Cooperative MoE
//
#include <pmmintrin.h>
#include <string.h>
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm/string.hpp>
#include <sys/time.h>
#include <stdlib.h>
#include "lexical_cast.hpp"

#include "config.h"

using namespace std;
using namespace boost;

class MOE
{
  private:
    int _n;
    int _k;

    float _alfa;
    float _beta;
    float _lamda1;
    float _lamda2;

    int _inter;
    string _input;
    string _model;
    string _validation;
    string _output;
    int _normalize;
    int _debug;
    int _thread_num;

	vector<vector<float> > _M;
    vector<vector<float> > _V;
    
    __m128 sse_alfa;
    __m128 sse_beta;
    __m128 sse_lamda1;
    __m128 sse_lamda2; 
    __m128 sse_one;
  
  public:
    MOE();
    ~MOE();
    void init(Config& config);
    float dot(vector<float>& w, vector<float>& xi, vector<int>& id);
    vector<float> g(vector<vector<float> >& w, vector<float>& xi, vector<int>& id);
    vector<float> w(vector<vector<float> >& v, vector<float>& xi, vector<int>& id);
    void vectorMultiply(vector<float>& a, vector<float>& b, vector<float>& r);
    void multiply(float val, vector<float>& b, vector<float>& r);
    void multiply(float val, vector<float>& a, vector<float>& b, vector<float>& r);
    void train(Config& config);
    void parser(string& line, vector<float>& x, int& y, vector<int>& id);
    float sigmoid(float wTx);
    void update(vector<float>& g_tmp, vector<float>& xi, vector<int>& id, vector<vector<float> >& z, vector<vector<float> >& m, int label);
    void updateSparse(vector<float>& g_tmp, vector<float>& xi, vector<int>& id, vector<vector<float> >& z, vector<vector<float> >& m, int label);
    void sign(float* in, float* out);
    float sign(float in);
	float sum(vector<float>& r);
    float logLoss(int& samples);
    float norm(vector<float>& x);
    void predict(Config& config);
    void saveModel();
    void loadModel();
    void debugMatrix(vector<vector<float> >& v, string s);
    void debugVector(vector<float>& v, string s);
	int64_t getTime();
};
