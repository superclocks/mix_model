#include <iostream>
#include <Eigen/Dense>
#include <vector>

#include<x86intrin.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>

using namespace std;
using namespace Eigen;
void threadTest()
{
  //#pragma omp parallel for num_threads(1)
  for(int i = 0; i < 2; i++)
  {
    #pragma omp parallel for num_threads(2)
    for(int j = 0; j < 4; j++)
    {
      printf("thread = %d, i = %d, j = %d\n", omp_get_thread_num(), i, j);
    }
  }
}
int main(int argc, char** argv)
{
  threadTest();
  Eigen::initParallel();
  
  if(argc < 3)
  {
    printf("Pls input the number of thread and the number of number.\n");
    exit(0);
  }
  int n_thread = atoi(argv[1]);
  struct timeval start, end;
  Eigen::setNbThreads(n_thread);
  int nn = Eigen::nbThreads();
  printf("Eigen thread = %d\n", nn);

  int M = 1;
  int N = 300000000;
  MatrixXf m = MatrixXf::Random(M, N);
  MatrixXf n = MatrixXf::Random(N, M); 
  
  VectorXd vec(N);
  for(int i = 0; i < N; i++)
	vec[i] = i;

  
  //Eigen计算矩阵乘法
  float timeuse;
  for(int t = 0; t < 1000; t++)
  {
  gettimeofday( &start, NULL ); 
  //MatrixXf r = m * n.transpose();
  double r = vec.dot(vec);
  printf("r = %f\n", r);
  gettimeofday( &end, NULL );
  
  timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
  printf("Eigen time: %f ms\n", timeuse / 1000.0);
  }
  //直接循环计算
  float r1 = 0.0;
  vector<float> v;
  for(int i = 0; i < N; i++)
    v.push_back(i);
  
  gettimeofday( &start, NULL );
  for(int i = 0; i < N; i++)
    r1 = r1 + v[i] * v[i];
  
  printf("r = %f\n", r1);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
  printf("sequence r: %f\n", r1);
  printf("sequence time: %f ms\n", timeuse / 1000.0);

  //OpenMP + SSE加速
  float* w = v.data();
  float res = 0;
  gettimeofday( &start, NULL );
  # pragma omp parallel for shared (w) num_threads(n_thread) reduction (+ : res)
  for(int d = 0; d < N; d += 4)
  {
    __m128 ai = _mm_load_ps(w + d);
    __m128 t = _mm_mul_ps(ai, ai);
    __m128 zero = _mm_setzero_ps();
    t = _mm_hadd_ps(t, zero);
    t = _mm_hadd_ps(t, zero);
    res += t[0];
    //res = res + t[0];
  }
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
  printf("openmp + sse r: %f\n", res);
  printf("openmp + sse time: %f ms\n", timeuse / 1000.0);

  /*
  std::cout << "The matrix m is of size "
            << m.rows() << "x" << m.cols() << std::endl;
  std::cout << "It has " << m.size() << " coefficients" << std::endl;
  VectorXd v(2);
  v.resize(5);
  std::cout << "The vector v is of size " << v.size() << std::endl;
  std::cout << "As a matrix, v is of size "
            << v.rows() << "x" << v.cols() << std::endl;
  */
  return 0;
}
