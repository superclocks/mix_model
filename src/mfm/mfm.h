#ifndef _MFM_
#define _MFM_
class MFM
{
  private:
    int _na; //广告类别
    int _naf; //广告特征个数
    int _nu; //用户类别
    int _nuf; //用户特征个数
    int _nc; //上下文类别
    int _ncf; //上下文特征个数
    int _k; //FM模型隐变量的维度

    float* _AW; //广告模型
    float* _UW; //用户模型
    float* _CW; //上下文模型
    float* _UV; //FM模型
  public:
    MFM();
    ~MFM();
    void init(Config& config);
};


#endif
