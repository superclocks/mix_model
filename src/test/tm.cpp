#include <stdlib.h>
#include <unordered_map>
#include <random>
#include <time.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm/string.hpp>
#include <stdlib.h>
#include "lexical_cast.hpp"
using namespace std;
using namespace boost;

void parser(string& line, vector<float>& x, int& y, vector<int>& id) 
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



void testIO()
{
  fstream f("../../data/WAP_xxl_PDPS000000057382.train", ios::in);
  if(!f.is_open())
  {
    cerr<< "cann't open file\n";
    exit(0);
  }
  vector<float> x(1000, 0);
  vector<int> id(1000, 0);
  int y;
  struct timeval start, end;
  gettimeofday( &start, NULL );
  string line;
  while(getline(f, line))
  {
    x.clear();
    id.clear();
    parser(line, x, y, id);
    //cout<< line << endl;
  }
  gettimeofday( &end, NULL );
  float timeuse = ( end.tv_sec - start.tv_sec ) * 1000 + (end.tv_usec -start.tv_usec) / 1000;
  printf("time: %f s\n", timeuse / 1000);
}
void testMap()
{

  unordered_map<string, int> dict;
  //default_random_engine generator(time(NULL)); 
  //uniform_real_distribution<float> dis(-1, 1); 
  char ss[3];
  printf("haha\n");
  for(int i = 0; i < 3; i++)
  {
    ss[0] = i + '0';
    for(int j = 0; j < 4; j++)
    {
      ss[1] = j + '0';
      for(int k = 0; k < 3; k++)
      {
        ss[2] = k + '0';
        printf("%d\t%d\t%d\t%s\t#\n", i, j, k, ss);
        //dict.insert(std::make_pair<std::string,int>(string(ss), i + j + k));
        dict[string(ss)] = i + j + k;
      }
    }
  }
  printf("hehe\n");
  unordered_map<string, int>::iterator it;
  for(it = dict.begin(); it != dict.end(); it++)
  {
    printf("%s\t%d\n", it->first.c_str(), it->second);
  }
}
void split(char* line)
{
  char* p = line;
  char* val = line;
  for(; *p != '\0'; p++)
  {
    if(*p == ' ')
    {
      *p = '\0';
      printf("%f\n", atof(val));
      *p = ' ';
      val = p + 1;
    }
  }
  printf("%f\n", atof(val));
}
int main(int argc, char** argv)
{

  char* line = new char[1024];
  strcpy(line, "123.34 3543.89");
  split(line);
  //testIO();
  /*
  default_random_engine generator;  
  uniform_real_distribution<double> dis(0,1);  
  string s = "haha";
  for(int i=0;i<5;i++)  
  {  
    std::cout<<dis(generator)<<std::endl;  
  }
  ifstream f("f.txt", ios::in);
  int k;
  f>>k;
  cout << k << endl;
  */
  return 0;
}








