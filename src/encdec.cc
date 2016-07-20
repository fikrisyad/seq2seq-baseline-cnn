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

using namespace std;
using namespace cnn;
using namespace cnn::expr;
using Sent = vector<int>;
using SentList = vector<Sent>;
using ParaSent = pair<Sent, Sent >;
using ParaCorp = vector<ParaSent >;
using BatchCol = vector<unsigned int>;
using Batch = vector<BatchCol>;
const int __LSTM__ = 0;
const int __FAST_LSTM__ = 1;
const int __GRU__ = 2;
const int __RNN__ = 3;

//parameters
int SOS_SRC;
int EOS_SRC;
int UNK_SRC;
int SOS_TRG;
int EOS_TRG;
int UNK_TRG;

template <class Builder>
struct AttentionalEncoderDecoder {
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_R;
  Parameters* p_bias;
  Parameters* p_Wa;
  Parameters* p_Ua;
  Parameters* p_va;
  Parameters* p_zero;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;
  Expression i_Uahj;
  Expression i_h_enc;
  unsigned int slen;
  boost::program_options::variables_map& vm;

  explicit AttentionalEncoderDecoder(Model& model, boost::program_options::variables_map& _vm) :
      dec_builder(
        _vm.at("depth-layer").as<unsigned int>(),
        (_vm.at("depth-layer").as<unsigned int>() * 2 + 1) * _vm.at("dim-hidden").as<unsigned int>(),
        _vm.at("dim-hidden").as<unsigned int>(),
        &model
      ),
      rev_enc_builder(
        _vm.at("depth-layer").as<unsigned int>(),
        _vm.at("dim-hidden").as<unsigned int>(),
        _vm.at("dim-hidden").as<unsigned int>(),
        &model
      ),
      fwd_enc_builder(
        _vm.at("depth-layer").as<unsigned int>(),
        _vm.at("dim-hidden").as<unsigned int>(),
        _vm.at("dim-hidden").as<unsigned int>(),
        &model
      ),
      vm(_vm) {
    vm = _vm;
    p_c = model.add_lookup_parameters(vm.at("src-vocab-size").as<unsigned int>(), {vm.at("dim-hidden").as<unsigned int>()}); 
    p_ec = model.add_lookup_parameters(vm.at("src-vocab-size").as<unsigned int>(), {vm.at("dim-hidden").as<unsigned int>()}); 
    p_R = model.add_parameters({vm.at("trg-vocab-size").as<unsigned int>(), vm.at("dim-hidden").as<unsigned int>()});
    p_bias = model.add_parameters({vm.at("trg-vocab-size").as<unsigned int>()});
    p_Wa = model.add_parameters({vm.at("dim-attention").as<unsigned int>(), unsigned(vm.at("dim-hidden").as<unsigned int>() * vm.at("depth-layer").as<unsigned int>())});
    p_Ua = model.add_parameters({vm.at("dim-attention").as<unsigned int>(), unsigned(vm.at("dim-hidden").as<unsigned int>() * 2 * vm.at("depth-layer").as<unsigned int>())});
    p_va = model.add_parameters({vm.at("dim-attention").as<unsigned int>()});
    p_zero = model.add_parameters({vm.at("dim-hidden").as<unsigned int>()});
  }

  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  void Encoder(const Batch sents, ComputationGraph& cg) {
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

  Expression Decoder(ComputationGraph& cg) {
    // decode
    Expression i_va = parameter(cg, p_va);
    Expression i_e_t = transpose(tanh(i_Uahj)) * i_va;
    Expression i_alpha_t = softmax(i_e_t);
    Expression i_c_t = i_h_enc * i_alpha_t;
    Expression i_init = parameter(cg, p_zero);
    Expression i_input = concatenate(vector<Expression>({i_init, i_c_t})); 
    Expression i_y_t = dec_builder.add_input(i_input);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;
    return i_r_t;
  }

  Expression Decoder(ComputationGraph& cg, const BatchCol prev) {
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
  
  void GreedyDecode(const Batch& sents, SentList& osents, ComputationGraph &cg){
   unsigned bsize = sents.at(0).size();
   //unsigned slen = sents.size();
   Encoder(sents, cg);
   Decoder(cg);
   Batch prev(1);
   osents.resize(bsize);
   for(unsigned int bi=0; bi < bsize; bi++){
     osents[bi].push_back(SOS_TRG);
     prev[0].push_back((unsigned int)SOS_TRG);
   }
   for (int t = 0; t < vm.at("limit-length").as<unsigned int>(); ++t) {
     unsigned int end_count = 0;
     for(unsigned int bi=0; bi < bsize; bi++){
       if(osents[bi][t] == EOS_TRG){
         end_count++;
       }
     }
     if(end_count == bsize) break;
     Expression i_r_t = Decoder(cg, prev[t]);
     Expression predict = softmax(i_r_t);
     vector<Tensor> results = cg.incremental_forward().batch_elems();
     prev.resize(t+2);
     for(unsigned int bi=0; bi < bsize; bi++){
       auto output = as_vector(results.at(bi));
       int w_id = 0;
       if(osents[bi][t] == EOS_TRG){
         w_id = EOS_TRG;
       }else{
         double w_prob = output[w_id];
         for(unsigned int j=0; j<output.size(); j++){
           double j_prob = output[j];
           if(j_prob > w_prob){
             w_id = j;
             w_prob = j_prob;
           }
         }
       }
       osents[bi].push_back(w_id);
       prev[t+1].push_back((unsigned int)w_id);
     }
   }
}

};

// Sort in descending order of length
struct CompareString {
  bool operator()(const ParaSent& first, const ParaSent& second) {
    if(
       (first.first.size() > second.first.size()) ||
       (first.first.size() == second.first.size() && first.second.size() > second.second.size())
    ){
      return true;
    }
    return false;
  }
  bool operator()(const Sent& first, const Sent& second) {
    if(first.size() > second.size()){
      return true;
    }
    return false;
  }
  bool operator()(const pair<string, unsigned int>& first, const pair<string, unsigned int>& second) {
    if(first.second > second.second){
      return true;
    }
    return false;
  }
};

void FreqCut(const string file_path, cnn::Dict& d, unsigned int dim_size){
  ifstream in(file_path);
  assert(in);
  map<string, unsigned int> str_freq;
  vector<pair<string, unsigned int> > str_vec;
  string line;
  while(getline(in, line)) {
    std::istringstream words(line);
    while(words){
      string word;
      words >> word;
      str_freq[word]++;
    }
  }
  in.close();
  for(auto& p1: str_freq){
   str_vec.push_back(pair<string, int>(p1.first, p1.second));
  }
  CompareString comp;
  sort(str_vec.begin(), str_vec.end(), comp);
  for(auto& p1 : str_vec){
    if(d.size() >= dim_size - 1){ // -1 for <UNK>
      break;
    }
    d.Convert(p1.first);
  }
}

void LoadCorpus(const string file_path, const int start, const int end, cnn::Dict& d, vector<Sent>& corpus){
  ifstream in(file_path);
  assert(in);
  int tlc = 0;
  int ttoks = 0;
  string line;
  while(getline(in, line)) {
    ++tlc;
    corpus.push_back(ReadSentence(line, &d));
    ttoks += corpus.back().size();
    if (corpus.back().front() != start && corpus.back().back() != end) {
      cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
      abort();
    }
  }
  in.close();
  cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
}

void ToBatch(const unsigned int bid, const unsigned int bsize, ParaCorp& sents, Batch& lbatch, Batch& rbatch){
  lbatch.resize(sents.at(bid).first.size());
  rbatch.resize(sents.at(bid).second.size());
  for(unsigned int sid = bid; sid< bid + bsize; sid++){
    for(unsigned int i=0; i<sents.at(bid).first.size(); i++){
      lbatch[i].push_back(sents.at(sid).first.at(i));
      rbatch[i].push_back(sents.at(sid).second.at(i));
    }
  }
}

void ToBatch(const unsigned int bid, const unsigned int bsize, SentList& sents, Batch& batch){
  batch.resize(sents.at(bid).size());
  for(unsigned int sid = bid; sid< bid + bsize; sid++){
    for(unsigned int i=0; i<sents.at(bid).size(); i++){
      batch[i].push_back(sents.at(bid).at(i));
    }
  }
}

void print_sent(Sent& osent, cnn::Dict& d_trg){
  for(auto wid : osent){
    string word = d_trg.Convert(wid);
    cout << word;
    if(wid == EOS_TRG){
      break;
    }
    cout << " ";
  }
  cout << endl;
}

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

template <class Builder>
void train(boost::program_options::variables_map& vm){
  cnn::Dict d_src, d_trg;
  ParaCorp training, dev;
  vector<Sent > training_src, training_trg, dev_src, dev_trg;
  SOS_SRC = d_src.Convert("<s>");
  EOS_SRC = d_src.Convert("</s>");
  cerr << "Reading source language training text from " << vm.at("path_train_src").as<string>() << "...\n";
  FreqCut(vm.at("path_train_src").as<string>(), d_src, vm.at("src-vocab-size").as<unsigned int>());
  d_src.Freeze(); // no new word types allowed
  d_src.SetUnk("<unk>");
  UNK_SRC = d_src.Convert("<unk>");
  //vm.at("src-vocab-size").as<int>() = d_src.size();
  LoadCorpus(vm.at("path_train_src").as<string>(), SOS_SRC, EOS_SRC, d_src, training_src);

  SOS_TRG = d_trg.Convert("<s>");
  EOS_TRG = d_trg.Convert("</s>");
  cerr << "Reading target language training text from " << vm.at("path_train_trg").as<string>() << "...\n";
  FreqCut(vm.at("path_train_trg").as<string>(), d_trg, vm.at("trg-vocab-size").as<unsigned int>());
  d_trg.Freeze(); // no new word types allowed
  d_trg.SetUnk("<unk>");
  UNK_TRG = d_trg.Convert("<unk>");
  //vm.at("trg-vocab-size").as<int>() = d_trg.size();
  LoadCorpus(vm.at("path_train_trg").as<string>(), SOS_TRG, EOS_TRG, d_trg, training_trg);
  cerr << "Writing source dictionary to " << vm.at("path_dict_src").as<string>() << "...\n";
  {
    ofstream out(vm.at("path_dict_src").as<string>());
    boost::archive::text_oarchive oa(out);
    oa << d_src;
    out.close();
  }
  cerr << "Writing target dictionary to " << vm.at("path_dict_trg").as<string>() << "...\n";
  {
    ofstream out(vm.at("path_dict_trg").as<string>());
    boost::archive::text_oarchive oa(out);
    oa << d_trg;
    out.close();
  }
  // for sorting
  for(unsigned int i = 0; i < training_src.size(); i++){
//cerr << i << " " << training_src.at(i).size() << " " << training_trg.at(i).size() << endl;
    ParaSent p(training_src.at(i), training_trg.at(i));
    training.push_back(p);
  }
  cerr << "creating mini-batches" << endl;
  CompareString comp;
  sort(training.begin(), training.end(), comp);
  for(unsigned int i = 0; i < training.size(); i += vm.at("batch-size").as<unsigned int>()){
    for(unsigned int  j = 1; j < vm.at("batch-size").as<unsigned int>() && i+j < training.size(); ++j){
      while(training.at(i+j).first.size() < training.at(i).first.size()){ // source padding
        training.at(i+j).first.push_back(EOS_SRC);
      }
      while(training.at(i+j).second.size() < training.at(i).second.size()){ // target padding
        training.at(i+j).second.push_back(EOS_TRG);
      }
    }
  }
  cerr << "Reading source development text from " << vm.at("path_dev_src").as<string>() << "...\n";
  LoadCorpus(vm.at("path_dev_src").as<string>(), SOS_SRC, EOS_SRC, d_src, dev_src);
  cerr << "Reading target development text from " << vm.at("path_dev_trg").as<string>() << "...\n";
  LoadCorpus(vm.at("path_dev_trg").as<string>(), SOS_TRG, EOS_TRG, d_trg, dev_trg);
  // for sorting
  for(unsigned int i=0; i < dev_src.size(); i++){
    ParaSent p(dev_src.at(i), dev_trg.at(i));
    dev.push_back(p);
  }
  // creating mini-batches
  sort(dev.begin(), dev.end(), comp);
  for(size_t i = 0; i < dev.size(); i += vm.at("batch-size").as<unsigned int>()){
    for(size_t j = 1; j < vm.at("batch-size").as<unsigned int>() && i+j < dev.size(); ++j){
      while(dev.at(i+j).first.size() < dev.at(i).first.size()){ // source padding
        dev.at(i+j).first.push_back(EOS_SRC);
      }
      while(dev.at(i+j).second.size() < dev.at(i).second.size()){ // target padding
        dev.at(i+j).second.push_back(EOS_TRG);
      }
    }
  }
  
  ostringstream os;
  os << "bilm"
     << '_' << vm.at("depth-layer").as<unsigned int>()
     << '_' << vm.at("dim-input").as<unsigned int>()
     << '_' << vm.at("dim-hidden").as<unsigned int>()
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;
  
  Model model;
  AttentionalEncoderDecoder<Builder> encdec(model, vm);
  Trainer* sgd = new SimpleSGDTrainer(&model);
  sgd->clip_threshold *= vm.at("batch-size").as<unsigned int>();
  
  vector<unsigned> order((training.size()+vm.at("batch-size").as<unsigned int>()-1)/vm.at("batch-size").as<unsigned int>());
  for (unsigned i = 0; i < order.size(); ++i){
    order[i] = i * vm.at("batch-size").as<unsigned int>();
  }

  vector<unsigned> dev_order((dev.size()+vm.at("batch-size").as<unsigned int>()-1)/vm.at("batch-size").as<unsigned int>());
  for (unsigned i = 0; i < dev_order.size(); ++i){
    dev_order[i] = i * vm.at("batch-size").as<unsigned int>();
  }

  unsigned lines = 0;
  while(1) {
    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);
    Timer iteration("completed in");
    double loss = 0;
    for (unsigned si = 0; si < order.size(); ++si) {
      // build graph for this instance
      ComputationGraph cg;
      unsigned bsize = std::min((unsigned)training.size() - order[si], vm.at("batch-size").as<unsigned int>()); // Batch size
      Batch sents, osents;
      ToBatch(order[si], bsize, training, sents, osents);
      encdec.Encoder(sents, cg);
      vector<Expression> errs;
      {
        Expression i_r_t = encdec.Decoder(cg);
        Expression i_err = pickneglogsoftmax(i_r_t, osents[0]);
      }
      for (int t = 0; t < osents.size() - 1; ++t) {
        Expression i_r_t = encdec.Decoder(cg, osents[t]);
        //vector<unsigned int> next = osents[t+1];
        Expression i_err = pickneglogsoftmax(i_r_t, osents[t+1]);
        errs.push_back(i_err);
      }
      Expression i_nerr = sum_batches(sum(errs));
      //cg.PrintGraphviz();
      loss += as_scalar(cg.forward()) / (double)bsize;
      cg.backward();
      sgd->update((1.0 / double(osents.at(0).size())));
      //sgd->update();
      cerr << " E = " << (loss / double(si + 1)) << " ppl=" << exp(loss / double(si + 1)) << ' ';
    }
    sgd->update_epoch();
    sgd->status();
    
#if 0
    lm.RandomSample();
#endif
    // show score on dev data?
    double dloss = 0;
    for(unsigned int dsi=0; dsi < dev_order.size(); dsi++){
      ComputationGraph cg;
      unsigned dev_bsize = std::min((unsigned)dev.size() - dev_order[dsi], vm.at("batch-size").as<unsigned int>()); // Batch size
      Batch isents, osents;
      SentList results;
      ToBatch(dev_order[dsi], dev_bsize, dev, isents, osents);
      encdec.GreedyDecode(isents, results, cg);
      for(unsigned int i = 0; i< results.size(); i++){
        dloss -= f_measure(dev.at(i + dev_order[dsi]).second, results.at(i), d_src, d_trg); // future work : replace to bleu
        cerr << "ref" << endl;
        print_sent(dev.at(i + dev_order[dsi]).second, d_trg);
        cerr << "hyp" << endl;
        print_sent(results.at(i), d_trg);
      }
    }
    if (dloss < best) {
      best = dloss;
      ofstream out(vm.at("path_model").as<string>());
      boost::archive::text_oarchive oa(out);
      oa << model;
      out.close();
    }
    ++lines;
    cerr << "\n***DEV [epoch=" << lines << "] F = " << (0 - dloss / (double)dev.size()) << ' ';
  }
  delete sgd;

}

template <class Builder>
void test(boost::program_options::variables_map& vm){
  cnn::Dict d_src, d_trg;
  ParaCorp training, dev;
  vector<Sent > test_src, test_out;
  cerr << "Reading source dictionary from " << vm.at("path_test_src").as<string>() << "...\n";
  {
    string fname = vm.at("path_dict_src").as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> d_src;
    in.close();
  }
  cerr << "Reading target dictionary from " << vm.at("path_test_src").as<string>() << "...\n";
  {
    string fname = vm.at("path_dict_trg").as<string>();
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> d_trg;
    in.close();
  }
  cerr << "Reading source test text from " << vm.at("path_test_src").as<string>() << "...\n";
  SOS_SRC = d_src.Convert("<s>");
  EOS_SRC = d_src.Convert("</s>");
  SOS_TRG = d_trg.Convert("<s>");
  EOS_TRG = d_trg.Convert("</s>");
  LoadCorpus(vm.at("path_test_src").as<string>(), SOS_SRC, EOS_SRC, d_src, test_src);
  //RNNBuilder rnn(vm.at("depth-layer").as<int>(), vm.at("dim-input").as<int>(), vm.at("dim-hidden").as<int>(), &model);
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  Model model;
  AttentionalEncoderDecoder<Builder> encdec(model, vm);
  string fname = vm.at("path_model").as<string>();
  cerr << "Reading model from " << vm.at("path_test_src").as<string>() << "...\n";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  in.close();
  // creating mini-batches
  CompareString comp;
  sort(test_src.begin(), test_src.end(), comp);
  for(size_t i = 0; i < test_src.size(); i += vm.at("batch-size").as<unsigned int>()){
    for(size_t j = 1; j < vm.at("batch-size").as<unsigned int>() && i+j < test_src.size(); ++j){
      while(test_src.at(i+j).size() < test_src.at(i).size()){ // source padding
        test_src.at(i+j).push_back(EOS_SRC);
      }
    }
  }
  vector<unsigned> test_order((test_src.size()+vm.at("batch-size").as<unsigned int>()-1)/vm.at("batch-size").as<unsigned int>());
  for (unsigned i = 0; i < test_order.size(); ++i){
    test_order[i] = i * vm.at("batch-size").as<unsigned int>();
  }
  for(unsigned int tsi=0; tsi < test_order.size(); tsi++){
    ComputationGraph cg;
    unsigned test_bsize = std::min((unsigned)test_order.size() - test_order[tsi], vm.at("batch-size").as<unsigned int>()); // Batch size
    Batch isents, osents;
    SentList results;
    ToBatch(test_order[tsi], test_bsize, test_src, isents);
    encdec.GreedyDecode(isents, results, cg);
    for(unsigned int i = 0; i < results.size(); i++){
      print_sent(test_src.at(i + test_order[tsi]), d_trg);
      print_sent(results.at(i), d_trg);
    }
  }
/*
  for(unsigned int i=0; i < test_src.size(); i++){
    ComputationGraph cg;
    vector<int> osent;
    greedy_decode(test_src.at(i), osent, encdec, cg);
    print_sent(osent);
    test_out.push_back(osent);
  }
*/
}

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description opts("h");
  opts.add_options()
  ("path_train_src", po::value<string>()->required(), "source train file")
  ("path_train_trg", po::value<string>()->required(), "target train file")
  ("path_dev_src", po::value<string>()->required(), "source dev file")
  ("path_dev_trg", po::value<string>()->required(), "target dev file")
  ("path_test_src", po::value<string>()->required(), "test input")
  ("path_test_out", po::value<string>()->required(), "test input")
  ("path_dict_src", po::value<string>()->required(), "source dictionary file")
  ("path_dict_trg", po::value<string>()->required(), "target dictionary file")
  ("path_model", po::value<string>()->required(), "test input")
  ("batch-size",po::value<unsigned int>()->default_value(1), "batch size")
  ("beam-size", po::value<unsigned int>()->default_value(1), "beam size")
  ("src-vocab-size", po::value<unsigned int>()->default_value(20000), "source vocab size")
  ("trg-vocab-size", po::value<unsigned int>()->default_value(20000), "target vocab size")
  ("builder", po::value<int>()->default_value(0), "select builder (0:LSTM (default), 1:Fast-LSTM, 2:GRU, 3:RNN)")
  ("train", po::value<int>()->default_value(1), "is training ? (1:Yes,0:No)")
  ("test", po::value<int>()->default_value(1), "is test ? (1:Yes, 0:No)")
  ("dim-input", po::value<unsigned int>()->default_value(500), "dimmension size of embedding layer")
  ("dim-hidden", po::value<unsigned int>()->default_value(500), "dimmension size of hidden layer")
  ("dim-attention", po::value<unsigned int>()->default_value(64), "dimmension size of hidden layer")
  ("depth-layer", po::value<unsigned int>()->default_value(1), "depth of hidden layer")
  ("limit-length", po::value<unsigned int>()->default_value(100), "length limit of target language in decoding")
  ("cnn-mem", po::value<string>()->default_value("512m"), "memory size");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);
  cnn::Initialize(argc, argv);
  if(vm.at("train").as<int>() > 0){
    switch(vm.at("builder").as<int>()){
      case __LSTM__:
      train<LSTMBuilder>(vm);
      break;
      case __FAST_LSTM__:
      train<FastLSTMBuilder>(vm);
      break;
      case __GRU__:
      train<GRUBuilder>(vm);
      break;
      case __RNN__:
      train<SimpleRNNBuilder>(vm);
      break;
    }
  }
  if(vm.at("test").as<int>() > 0){
    switch(vm.at("builder").as<int>()){
      case __LSTM__:
      test<LSTMBuilder>(vm);
      break;
      case __FAST_LSTM__:
      test<FastLSTMBuilder>(vm);
      break;
      case __GRU__:
      test<GRUBuilder>(vm);
      break;
      case __RNN__:
      test<SimpleRNNBuilder>(vm);
      break;
    }
  }
}
