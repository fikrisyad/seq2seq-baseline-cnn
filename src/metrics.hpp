#include "cnn/dict.h"
#include "define.hpp"

#ifndef INCLUDE_GUARD_METRICS_HPP
#define INCLUDE_GUARD_METRICS_HPP

using namespace std;
using namespace cnn;
using namespace cnn::expr;

// BLEU is the best, but mendoi
double f_measure(Sent &isent, Sent &osent, cnn::Dict& d_src, cnn::Dict& d_trg){
  std::map<string, bool> word_src;
  std::map<string, bool> word_trg;
  for(auto i : isent){
    word_src[d_src.Convert(i)] = true;
  }
  double cnt = 0;
  for(auto o : osent){
    word_trg[d_trg.Convert(o)] = true;
    if(word_src.count(d_trg.Convert(o)) > 0){
      cnt++;
    }
  }
  if(cnt == 0){
    return 0.0;
  }
  double p = (double)(cnt) / (double)(osent.size());
  cnt = 0;
  for(auto i : isent){
    if(word_trg.count(d_src.Convert(i)) > 0){
      cnt++;
    }
  }
  if(cnt == 0){
    return 0.0;
  }
  int r = (double)(cnt) / (double)(isent.size());
  double f = 2.0 * p * r / (p + r);
  return f;
}

#endif
