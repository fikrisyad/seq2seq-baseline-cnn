#include "cnn/dict.h"
#include "define.hpp"

#ifndef INCLUDE_GUARD_METRICS_HPP
#define INCLUDE_GUARD_METRICS_HPP

using namespace std;
using namespace cnn;
using namespace cnn::expr;

// BLEU is the best, but mendoi
double f_measure(Sent &output, Sent &ref, cnn::Dict& d_src, cnn::Dict& d_trg){
  std::map<string, bool> word_src;
  std::map<string, bool> word_trg;
  for(auto i : output){
    word_src[d_src.Convert(i)] = true;
  }
  double cnt = 0;
  for(auto o : ref){
    word_trg[d_trg.Convert(o)] = true;
    if(word_src.count(d_trg.Convert(o)) > 0){
      cnt++;
    }
  }
  if(cnt == 0){
    return 0.0;
  }
  double p = (double)(cnt) / (double)(ref.size());
  cnt = 0;
  for(auto i : output){
    if(word_trg.count(d_trg.Convert(i)) > 0){
      cnt++;
    }
  }
  if(cnt == 0){
    return 0.0;
  }
  int r = (double)(cnt) / (double)(output.size());
  double f = 2.0 * p * r / (p + r);
  return f;
}

#endif
