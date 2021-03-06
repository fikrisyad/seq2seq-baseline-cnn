#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/fast-lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include "define.hpp"
#include "encdec.hpp"

#ifndef INCLUDE_GUARD_Bahdanau2014_HPP
#define INCLUDE_GUARD_Bahdanau2014_HPP

using namespace std;
using namespace cnn;
using namespace cnn::expr;

template <class Builder>
class Bahdanau2014 : public EncoderDecoder<Builder> {

public:
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_R;
  Parameters* p_bias;
  Parameters* p_Wa;
  Parameters* p_Ua;
  Parameters* p_va;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;
  Expression i_Uahj;
  Expression i_h_enc;
  unsigned int slen;
  boost::program_options::variables_map* vm;

  explicit Bahdanau2014(Model& model, boost::program_options::variables_map* _vm) :
    dec_builder(
      _vm->at("depth-layer").as<unsigned int>(),
      (_vm->at("depth-layer").as<unsigned int>() * 2 + 1) * _vm->at("dim-hidden").as<unsigned int>(),
      _vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    rev_enc_builder(
      _vm->at("depth-layer").as<unsigned int>(),
      _vm->at("dim-hidden").as<unsigned int>(),
      _vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    fwd_enc_builder(
      _vm->at("depth-layer").as<unsigned int>(),
      _vm->at("dim-hidden").as<unsigned int>(),
      _vm->at("dim-hidden").as<unsigned int>(),
      &model
    ),
    vm(_vm)
  {
    p_c = model.add_lookup_parameters(vm->at("trg-vocab-size").as<unsigned int>(), {vm->at("dim-hidden").as<unsigned int>()}); 
    p_ec = model.add_lookup_parameters(vm->at("src-vocab-size").as<unsigned int>(), {vm->at("dim-hidden").as<unsigned int>()}); 
    p_R = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>(), vm->at("dim-hidden").as<unsigned int>()});
    p_bias = model.add_parameters({vm->at("trg-vocab-size").as<unsigned int>()});
    p_Wa = model.add_parameters({vm->at("dim-attention").as<unsigned int>(), unsigned(vm->at("dim-hidden").as<unsigned int>() * vm->at("depth-layer").as<unsigned int>())});
    p_Ua = model.add_parameters({vm->at("dim-attention").as<unsigned int>(), unsigned(vm->at("dim-hidden").as<unsigned int>() * 2 * vm->at("depth-layer").as<unsigned int>())});
    p_va = model.add_parameters({vm->at("dim-attention").as<unsigned int>()});
  }

  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  virtual void Encoder(const Batch sents, ComputationGraph& cg) {
    // forward encoder
    slen = sents.size();
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    vector<Expression> h_fwd(sents.size());
    vector<Expression> h_bwd(sents.size());
    vector<Expression> h_bi(sents.size());
    for (unsigned i = 0; i < sents.size(); ++i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      //h_fwd[i] = fwd_enc_builder.add_input(i_x_t);
      fwd_enc_builder.add_input(i_x_t);
      h_fwd[i] = concatenate(fwd_enc_builder.final_h());
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int i = sents.size() - 1; i >= 0; --i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      //h_bwd[i] = rev_enc_builder.add_input(i_x_t);
      rev_enc_builder.add_input(i_x_t);
      h_bwd[i] = concatenate(rev_enc_builder.final_h());
    }
    // bidirectional encoding
    for (unsigned i = 0; i < sents.size(); ++i) {
      h_bi[i] = concatenate(vector<Expression>({h_fwd[i], h_bwd[i]}));
    }
    i_h_enc = concatenate_cols(h_bi);
    Expression i_Ua = parameter(cg, p_Ua);
    i_Uahj = i_Ua * i_h_enc;
    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(rev_enc_builder.final_s());
  }

  virtual Expression Decoder(ComputationGraph& cg, const BatchCol prev) {
    // decode
    Expression i_va = parameter(cg, p_va);
    Expression i_Wa = parameter(cg, p_Wa);
    Expression i_h_prev = concatenate(dec_builder.final_h());
    Expression i_wah = i_Wa * i_h_prev;
    Expression i_Wah = concatenate_cols(vector<Expression>(slen, i_wah));
    Expression i_e_t = transpose(tanh(i_Wah + i_Uahj)) * i_va;
    Expression i_alpha_t = softmax(i_e_t);
    Expression i_c_t = i_h_enc * i_alpha_t;

    Expression i_x_t = lookup(cg, p_c, prev);
    Expression input = concatenate(vector<Expression>({i_x_t, i_c_t})); 
    Expression i_y_t = dec_builder.add_input(input);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;
    return i_r_t;
  }
  
};

#endif // INCLUDE_GUARD_HOGE_HPP
