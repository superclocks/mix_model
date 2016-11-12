#ifndef _CONFIG_
#define _CONFIG_
#include <string>
using namespace std;

class Config
{
public:
    int _n;
    int _k;

    float _alfa;
    float _beta;
    float _lamda1;
    float _lamda2;

    int _inter;
    string _input; //train file path or predict file path
    string _model; //model file path
    string _validation; //validation file path
    string _output; //predict result file path
    int _mode; //train or predict
    int _normalize;
	int _debug;
	int _thread_num;
};
#endif

