#include "moe.h"

void MOE::init(Config& config)
{
  //初始化参数
  _input = config._input;
  _model = config._model;
  _validation = config._validation;
  _output = config._output;
  _n = config._n + 1;
  _k = config._k;
  _alfa = config._alfa;
  _beta = config._beta;
  _lamda1 = config._lamda1;
  _lamda2 = config._lamda2;
  _inter = config._inter;
  _normalize = config._normalize;
  _debug = config._debug;
  _thread_num = config._thread_num;
  //
  int nt = _n % 2;
  _n += nt;
  //初始化模型_M和_V
  default_random_engine generator(time(NULL)); 
  uniform_real_distribution<float> dis(-1, 1);
  vector<float> zero(_n, 0.0);
  vector<vector<float> > zeros(_k, zero);
  _M = zeros;
  _V = zeros;
  dis(generator);
  for(int i = 0; i < _k; i++)
  {
    for(int j = 0; j < _n - nt; j++)
	{
      _V[i][j] = dis(generator) * 0.0001;
	  _M[i][j] = dis(generator) * 0.0001;
	}
  }
  /*
  if(_debug == 1)
  {
    ifstream mv("../data/mv.txt", ios::in);
    for(int i = 0; i < 4; i++)
    {
	  for(int j = 0; j < 3; j++)
	  {
	    float tmp;
	    mv>> tmp;
	    _M[i][j] = tmp;
	  }
    }
    for(int i = 0; i < 4; i++)
    {
	  for(int j = 0; j < 3; j++)
	  {
	    float tmp;
	    mv>> tmp;
	    _V[i][j] = tmp;
	  }
    }
    mv.close();
  }
  */
  //
  sse_alfa = _mm_set1_ps(_alfa);
  sse_beta = _mm_set1_ps(_beta);
  sse_lamda1 = _mm_set1_ps(_lamda1);
  sse_lamda2 = _mm_set1_ps(_lamda2); 
  sse_one = _mm_set1_ps(-1.0);
}
MOE::MOE()
{
}
MOE::~MOE()
{

}
float MOE::dot(vector<float>& w, vector<float>& xi, vector<int>& id)
{
  size_t n = id.size();
  float r = 0.0;
  for(size_t i = 0; i < n; i++)
  {
    int index = id[i];
    r += w[index] * xi[i];
  }
  return r;
}

vector<float> MOE::g(vector<vector<float> >& w, vector<float>& xi, vector<int>& id)
{
  int n = w.size();
  float sum = 0.0;
  vector<float> r;
  for(int i = 0; i < n; i++)
  {
    float d = exp(dot(w[i], xi, id));
    r.push_back(d);
    sum += d;
  }
  for(int i = 0; i < n; i++)
  {
    r[i] /= sum;
  }
  return r;
}
void MOE::debugMatrix(vector<vector<float> >& v, string s)
{
  cout<< s << endl;
  for(size_t i = 0; i < v.size(); i++)
  {
    for(size_t j = 0; j < v[i].size(); j++)
    {
      cout<< v[i][j] << " ";
    }
    cout<< endl;
  }
}
void MOE::debugVector(vector<float>& v, string s)
{
  cout<<s << endl;
  for(size_t i = 0; i < v.size(); i++)
  {
	cout<< v[i] << " ";
  }
  cout<< endl;
}
vector<float> MOE::w(vector<vector<float> >& v, vector<float>& xi, vector<int>& id)
{
  int n = v.size();
  vector<float> r;
  for(int i = 0; i < n; i++)
  {
    r.push_back(dot(v[i], xi, id));
  }
  return r;
}

void MOE::vectorMultiply(vector<float>& a, vector<float>& b, vector<float>& r)
{
  size_t a_n = a.size();
  size_t b_n = b.size();
  if(a_n != b_n || a_n == 0 || b_n == 0)
  {
	cerr<<"the size of vector in vectorMultiply must have same length"<< endl;
	exit(0);
  }
  for(size_t i = 0; i < a_n; i++)
  {
	r[i] = a[i] * b[i];
  }
  /*
  int m = n / 4;
  float* pa = a.data();
  float* pb = b.data();
  float* pr = r.data();
  for(int i = 0; i < m; i++)
  {
    __m128 sse_a = _mm_load_ps(pa + i * 4);
    __m128 sse_b = _mm_load_ps(pb + i * 4);
    __m128 mul = _mm_mul_ps(sse_a, sse_b);
    _mm_store_ps(pr + i * 4, mul);
  }
  for(int i = m * 4; i < n; i++)
  {
    r[i] = a[i] * b[i];
  }
  */
}
void MOE::multiply(float val, vector<float>& b, vector<float>& r)
{
  size_t n = b.size();
  for(size_t i = 0; i < n; i++)
  {
	r[i] = -val * b[i];
  }
  /*
  int m = n / 4;
  float* pb = b.data();
  float* pr = r.data();
  __m128 sse_val = _mm_set1_ps(val);
  for(int i = 0; i < m; i++)
  {
    __m128 sse_b = _mm_load_ps(pb + i * 4);
    __m128 mul = -_mm_mul_ps(sse_val, sse_b);
    _mm_store_ps(pr + i * 4, mul);
  }
  for(int i = m * 4; i < n; i++)
  {
    r[i] = -val * b[i];
  }
  */
}

void MOE::multiply(float val, vector<float>& a, vector<float>& b, vector<float>& r)
{
  size_t n = a.size();
  for(size_t i = 0; i < n; i++)
  {
	r[i] = -val * (a[i] - a[i] * b[i]) * b[i];
  }
  /*
  int m = n / 4;
  float* pa = a.data();
  float* pb = b.data();
  float* pr = r.data();
  __m128 sse_val = _mm_set1_ps(val); 

  for(int i = 0; i < m; i++)
  {
    __m128 sse_a = _mm_load_ps(pa + i * 4);
    __m128 sse_b = _mm_load_ps(pb + i * 4);
    __m128 mul = -_mm_mul_ps(sse_val, _mm_mul_ps(_mm_sub_ps(sse_a, _mm_mul_ps(sse_a, sse_b)), sse_b));
    _mm_store_ps(pr + i * 4, mul);
  }
  for(int i = m * 4; i < n; i++)
  {
    r[i] = -val * (a[i] - a[i] * b[i]) * b[i];
  }
  */
}
void MOE::update(vector<float>& g_tmp, vector<float>& xi, vector<int>& id, vector<vector<float> >& z, vector<vector<float> >& n, int label)
{
  size_t tt = xi.size();
  //if (label == 1)
  	//debugMatrix(z, "original_z");
  //cout << "signa" << endl;
  for(int i = 0; i < _k; i++)
  {
    for(size_t j = 0; j < tt; j++)
    {
      float dvij = g_tmp[i] * xi[j];
      float sigma = (sqrt(n[i][id[j]] + dvij * dvij) - sqrt(n[i][id[j]])) / _alfa;
      if(label == 0)
	    z[i][id[j]] += dvij - sigma * _V[i][id[j]];
	  else if(label == 1)
		z[i][id[j]] += dvij - sigma * _M[i][id[j]];
	
      n[i][id[j]] += dvij * dvij;
	  //cout<< sigma << " " << dvij << " " << "dm = " <<dvij << "sigma = "<< sigma << "Vij = " << _V[i][id[j]] << " ";
    }
	//cout<< endl;
  }
	
  int t = _n / 4;
  //#pragma omp parallel for num_threads(_k + 4)
  for(int i = 0; i < _k; i++)
  {
    float* np = n[i].data();
    float* zp = z[i].data();
    float* para = 0;
    if(label == 0)
    {
      para = _V[i].data();
    }
    else if(label == 1)
    {
      para = _M[i].data();
    }
    float sign_z[4];

//#pragma omp parallel for num_threads(10)
	for(int j = 0; j < t; j++)
    {
      __m128 sse_n = _mm_load_ps(np + j * 4);
      __m128 sse_z = _mm_load_ps(zp + j * 4);
      sign(zp + j * 4, sign_z);
      __m128 sse_sign_z = _mm_load_ps(sign_z);

      __m128 sse_a = _mm_div_ps(sse_one, _mm_add_ps(_mm_div_ps(_mm_add_ps(sse_beta, _mm_sqrt_ps(sse_n)), sse_alfa), sse_lamda2));
      __m128 sse_b = _mm_sub_ps(sse_z, _mm_mul_ps(sse_sign_z, sse_lamda1));
      __m128 sse_ab = _mm_mul_ps(sse_a, sse_b);
      _mm_store_ps(para + j * 4, sse_ab);  
      for(int k = 0; k < 4; k++)
      {
        if(abs(*(zp + i * 4 + k)) < _lamda1)
          *(para + i * 4 + k) = 0.0;
      }
    }
  }
}
float MOE::sign(float in)
{
  if(in > 0.0)
    return 1.0;
  else if(in < 0.0)
    return -1.0;
  else
    return 0.0;
}
void MOE::updateSparse(vector<float>& g_tmp, vector<float>& xi, vector<int>& id, vector<vector<float> >& z, vector<vector<float> >& n, int label)
{
  size_t tt = xi.size();
  //if (label == 1)
  	//debugMatrix(z, "original_z");
  //cout << "signa" << endl;
  #pragma omp parallel for num_threads(_thread_num)
  for(int i = 0; i < _k; i++)
  {
    for(size_t j = 0; j < tt; j++)
    {
      float dvij = g_tmp[i] * xi[j];
      float sigma = (sqrt(n[i][id[j]] + dvij * dvij) - sqrt(n[i][id[j]])) / _alfa;
      if(label == 0)
        z[i][id[j]] += dvij - sigma * _V[i][id[j]];
      else if(label == 1)
        z[i][id[j]] += dvij - sigma * _M[i][id[j]];

      n[i][id[j]] += dvij * dvij;
      
      float weight = -1.0 / ( (_beta + sqrt(n[i][id[j]]))/ _alfa + _lamda2) * (z[i][id[j]] - sign(z[i][id[j]]) * _lamda1);
      if(label == 0)
      {
        if((abs(z[i][id[j]]) <= _lamda1))
          _V[i][id[j]] = 0.0;
        else
          _V[i][id[j]] = weight;
      }
      else if(label == 1)
      {
        if((abs(z[i][id[j]]) <= _lamda1))
          _M[i][id[j]] = 0.0;
        else
          _M[i][id[j]] = weight;
      }
      //cout<< sigma << " " << dvij << " " << "dm = " <<dvij << "sigma = "<< sigma << "Vij = " << _V[i][id[j]] << " ";
    }
  //cout<< endl;
  }

}

void MOE::sign(float* in, float* out)
{
  for(int i = 0; i < 4; i++)
  {
    if(in[i] > 0.0)
      out[i] = 1.0;
    else if(in[i] == 0)
      out[i] = 0.0;
    else
      out[i] = -1.0;
  }
}

float MOE::logLoss(int& samples)
{
  vector<float> r(_M.size(), 0); 
  vector<float> x;
  vector<int> id;
  string line;
  int y = 0;
  float loss = 0.0;
  int n = 0;
  ifstream f(_validation, ios::in);
  if(!f.is_open())
  {
    cerr<< "Can not open validation file" << endl;
    exit(0);
  }
  while(getline(f, line))
  {
    x.clear();
    id.clear();
    parser(line, x, y, id);
	if(_debug == 1)
	  debugVector(x, "x");
	if(_normalize == 1)
	  norm(x);
    //Gating Model
    vector<float> g_vector = g(_M, x, id);
    //Expert Model
    vector<float> w_vector = w(_V, x, id);
    vectorMultiply(g_vector, w_vector, r);
    float p = sigmoid(sum(r));
    //loss += y * log2f(p) + (1.0 - y) * log2f(1.0 - p);
    loss += y * log(p) + (1.0 - y) * log(1.0 - p);
    n++;
  }
  f.close();
  samples = n;
  return -loss / n;
}

void MOE::predict(Config& config)
{
  init(config); //init paras
  loadModel(); //load model

  vector<float> r(_M.size(), 0); 
  vector<float> x;
  vector<int> id;
  string line;
  int y;
  ifstream f_i(_input, ios::in);
  ofstream f_o(_output, ios::out);
  if(!f_i.is_open())
  {
    cerr<< "Can not open predict file" << endl;
    exit(0);
  }
  if(!f_o.is_open())
  {
    cerr<< "Can not open predict result file" << endl;
    exit(0);
  }
  while(getline(f_i, line))
  {
    x.clear();
    id.clear();
    parser(line, x, y, id);
	float weight;
	if(_normalize == 1)
	  weight = norm(x);
    //Gating Model
    vector<float> g_vector = g(_M, x, id);
    //Expert Model
    vector<float> w_vector = w(_V, x, id);
    vectorMultiply(g_vector, w_vector, r);
    float p = sigmoid(sum(r));
	/*int y_pred;
	if(p >= 0.5)
	  y_pred = 1;
	else
	  y_pred = 0;
	*/
    f_o<<y <<" " <<p <<endl;
	//f_o<< y_pred <<" " <<x[1] * weight <<" " <<x[2] * weight << endl;
  }
  f_i.close();
  f_o.close();
}

void MOE::train(Config& config)
{
  init(config); //init paras
  
  int run_num = 0;
  vector<int> id;
  vector<float> r(_M.size(), 0); //
  vector<float> g_vec_tmp(_k, 0);
  vector<float> w_vec_tmp(_k, 0);
  vector<vector<float> > dv;
  vector<vector<float> > dm;
  vector<vector<float> > z_v;
  vector<vector<float> > z_m;
  vector<vector<float> > n_v;
  vector<vector<float> > n_m;

  for(int i = 0; i < _k; i ++)
  {
    vector<float> t(_n, 0);
    dv.push_back(t);
    dm.push_back(t);
    z_v.push_back(t);
    z_m.push_back(t);
    n_v.push_back(t);
    n_m.push_back(t);
  }
  if(_debug == 1)
  {
    debugMatrix(_V, "original_V");
    debugMatrix(_M, "original_M");
  }
  while(run_num < _inter)
  {
	 int64_t tm = getTime();

     ifstream f(_input.c_str(), ios::in);
     if(!f.is_open())
     {
       cerr<< "Cant not open train file" << endl;
       exit(0);
     }
     string line;
     vector<float> x;
     int y;
     int samples = 0;
     ios::sync_with_stdio(false);
     while(getline(f, line))
     {
       x.clear();
       id.clear();
       parser(line, x, y, id);
	   if(_debug == 1)
	     cout<<"===========" << samples << "============" << endl;
       //Normalize
       if(_normalize == 1)
         norm(x);
       if(_debug == 1)
	     debugVector(x, "x");
	   //Gating Model
       vector<float> g_vector = g(_M, x, id);
       if(_debug == 1)
	     debugVector(g_vector, "g_vector"); 
       //Expert Model
       vector<float> w_vector = w(_V, x, id);
       if(_debug == 1)
	     debugVector(w_vector, "w_vector");
       vectorMultiply(g_vector, w_vector, r);
       
       float y_pred = sigmoid(sum(r));
       float delta = y - y_pred;
       if(_debug == 1)
	     cout<< "delta = " << delta << endl;
       //=============computer gradient   ==============
       multiply(delta, g_vector, g_vec_tmp);
       if(_debug == 1)
	     debugVector(g_vec_tmp, "g_vec_tmp");
	   multiply(delta, w_vector, g_vector, w_vec_tmp);
	   if(_debug == 1)
	     debugVector(w_vec_tmp, "w_vec_tmp");
       //=============update _V using FTRL==============
       updateSparse(g_vec_tmp, x, id, z_v, n_v, 0); 
       if(_debug == 1)
	     debugMatrix(_V, "V");
       //=============update _M using FTRL==============
       updateSparse(w_vec_tmp, x, id, z_m, n_m, 1);
       if(_debug == 1)
	   {
		 debugMatrix(_M, "M");
         cout<<"===========" << samples << "============" << endl;
	   }
       samples++;
     }
     f.close();
     float logloss = 0.0;
     int test_samples = 0;
     if(_validation.compare("") != 0)
     {
       logloss = logLoss(test_samples);
     }
     cout<<"#" <<run_num << " train samples = " << samples << " test samples = " <<test_samples <<" logloss = " << logloss <<" elapsed time = " << getTime() - tm <<"s" << endl;
     run_num++;
  }
}
float MOE::sum(vector<float>& r)
{
  float num = 0.0;
  size_t n = r.size();
  for(size_t i = 0; i < n; i++)
  {
    num += r[i];
  }
  return num;
}
float MOE::sigmoid(float wTx)
{
  if(wTx > -28.0 && wTx < 28.0)
    return exp(wTx) / (1.0 + exp(wTx));
  else if(wTx <= -28.0)
    return exp(-28.0) / (1.0 + exp(-28.0));
  else
    return exp(28.0) / (1.0 + exp(28.0));
}
float MOE::norm(vector<float>& x)
{
  float r = 0.0;
  size_t n = x.size();
  for(size_t i = 1; i < n; i++)
  {
    r += x[i] * x[i];
  }
  r = sqrt(r);
  for(size_t i = 1; i < n; i++)
  {
    x[i] /= r;
  }
  return r;
}

void MOE::parser(string& line, vector<float>& x, int& y, vector<int>& id)
{
  x.push_back(1.0);
  id.push_back(0);

  string str = line; //.substr(0, line.length() - 1);
  vector<string> vec;
  split(vec, str, is_any_of(" "));
  y = lexical_cast<int>(vec[0]);
  int n = vec.size();
  for(int i = 1; i < n; i++)
  {
    vector<string> tmp;
    split(tmp, vec[i], is_any_of(":"));
    if(tmp.size() == 2)
    {
      //id.push_back(lexical_cast<int>(tmp[1]) + 1);
	  id.push_back(atoi(tmp[1].c_str()) + 1);
      x.push_back(1.0);

    }
    else if(tmp.size() == 3)
    {
      //id.push_back(lexical_cast<int>(tmp[1]) + 1);
      //x.push_back(lexical_cast<float>(tmp[2]));
	  id.push_back(atoi(tmp[1].c_str()) + 1);
	  x.push_back(atof(tmp[2].c_str()));
    }
	
  }
}

void MOE::saveModel()
{
  ofstream f(_model, ios::out);
  if(!f.is_open())
  {
    cerr<< "Can not open model file wher saveModel" << endl;
    exit(0);
  }
  f<<"n " << _n << endl;
  f<<"k " << _k << endl;
  size_t m = _V.size();
  size_t n = _V[0].size();
  f<<"V:" << endl;
  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
    {
      f<< _V[i][j];
      if(j < n - 1)
    f<<" ";
    }
    f<<endl;
  }

  f<<"M:" << endl;

  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
    {
      f<< _M[i][j];
      if(j < n - 1)
    f<<" ";
    }
    f<<endl;
  }
  f.close();
}

void MOE::loadModel()
{
  ifstream f(_model, ios::in);
  if(!f.is_open())
  {
    cerr<< "Can not open model file when loadModel" << endl;
  }
  int n, k;
  string tmp;
  f>> tmp;
  f>> n;
  f>> tmp;
  f>> k;
  _n = n;
  _k = k;
  for(int i = 0; i < k; i++)
  {
    vector<float> tmp(n, 0.0);
    _V.push_back(tmp);
  }
  _M = _V;

  f>> tmp;
  //laod V
  for(int i = 0; i < k; i++)
  {
    for(int j = 0; j < n; j++)
    {
      float w;
      f>>w;
      _V[i][j] = w;
    }
  }
  //load M
  f>> tmp;
  for(int i = 0; i < k; i++)
  {
    for(int j = 0; j < n; j++)
    {
      float w;
      f>>w;
      _M[i][j] = w;
    }
  }
  f.close();
}

int64_t MOE::getTime()
{
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<int64_t>((now.tv_sec * 1000 + now.tv_usec / 1000) / 1000);
}
