#include "mfm.h"


MOE::MOE()
{}

MOE::~MOE()
{
    delete[] _AW;
    delete[] _UW;
    delete[] _CW;
    delete[] _UV;
}

void MOE::init(Config& config)
{
    _na = config._na;
    _naf = config._naf + 1;
    _nu = config._nu;
    _nuf = config._nuf + 1;
    _nc = config._nc;
    _ncf = config._ncf + 1;
    _k = config._k;
    _AW = new float[_na * _naf];
    _UW = new float[_nu * _nuf];
    _CW = new float[_nc * _ncf];
    _UV = 

}
