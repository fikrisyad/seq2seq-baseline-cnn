#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/fast-lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

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

//parameters
unsigned LAYERS = 3;
unsigned INPUT_DIM = 500;
unsigned HIDDEN_DIM = 500;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;
unsigned BATCH_SIZE = 1;
unsigned BEAM_SIZE = 1;
unsigned LENGTH_LIMIT = 100;
unsigned TRAIN = 0;
unsigned TEST = 0;
string PATH_TRAIN_SRC = "";
string PATH_TRAIN_TRG = "";
string PATH_DEV_SRC = "";
string PATH_DEV_TRG = "";
string PATH_TEST_SRC = "";
string PATH_TEST_OUT = "";
string PATH_DICT_SRC = "";
string PATH_DICT_TRG = "";
string PATH_MODEL = "";

cnn::Dict d_src, d_trg;
ParaCorp training, dev;
vector<Sent > training_src, training_trg, dev_src, dev_trg, test_src, test_out;
int SOS_SRC;
int EOS_SRC;
int UNK_SRC;
int SOS_TRG;
int EOS_TRG;
int UNK_TRG;

template <class Builder>
struct EncoderDecoder {
  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_ie2h;
  Parameters* p_bie;
  Parameters* p_h2oe;
  Parameters* p_boe;
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;
  explicit EncoderDecoder(Model& model) :
      dec_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {


    p_ie2h = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * 2)});
    p_bie = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    p_h2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)});
    p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS)});
    p_c = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
    p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({OUTPUT_VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({OUTPUT_VOCAB_SIZE});
  }

  // build graph and return Expression for total loss
  //void BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
  void Encoder(const Batch sents, ComputationGraph& cg) {
    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (const auto input : sents) {
      Expression i_x_t = lookup(cg, p_ec, input);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int i = sents.size() - 1; i >= 0; --i) {
      Expression i_x_t = lookup(cg, p_ec, sents[i]);
      rev_enc_builder.add_input(i_x_t);
    }
    
    // encoder -> decoder transformation
    vector<Expression> to;
    for (auto h_l : fwd_enc_builder.final_h()) to.push_back(h_l);
    for (auto h_l : rev_enc_builder.final_h()) to.push_back(h_l);
    
    Expression i_combined = concatenate(to);
    Expression i_ie2h = parameter(cg, p_ie2h);
    Expression i_bie = parameter(cg, p_bie);
    Expression i_t = i_bie + i_ie2h * i_combined;
    cg.incremental_forward();
    Expression i_h = rectify(i_t);
    Expression i_h2oe = parameter(cg,p_h2oe);
    Expression i_boe = parameter(cg,p_boe);
    Expression i_nc = i_boe + i_h2oe * i_h;
    
    vector<Expression> oein1, oein2, oein;
    for (unsigned i = 0; i < LAYERS; ++i) {
      oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
      oein2.push_back(tanh(oein1[i]));
    }
    for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
    for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(oein);
  }

  Expression Decoder(ComputationGraph& cg) {
    // decode
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * dec_builder.final_h().back();
    return i_r_t;
  }

  Expression Decoder(ComputationGraph& cg, const BatchCol prev) {
    // decode
    Expression i_x_t = lookup(cg, p_c, prev);
    Expression i_y_t = dec_builder.add_input(i_x_t);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;
    return i_r_t;
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
    if (training_src.back().front() != start && training_src.back().back() != end) {
      cerr << "Sentence in " << file_path << ":" << tlc << " didn't start or end with <s>, </s>\n";
      abort();
    }
  }
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

void print_sent(Sent& osent){
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
double f_measure(Sent &isent, Sent &osent){
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
void greedy_decode(const Batch& sents, SentList& osents, EncoderDecoder<Builder> &encdec, ComputationGraph &cg){
   unsigned bsize = sents.at(0).size();
   unsigned slen = sents.size();
   encdec.Encoder(sents, cg);
   encdec.Decoder(cg);
   Batch prev(1);
   osents.resize(bsize);
   for(unsigned int bi=0; bi < bsize; bi++){
     osents[bi].push_back(SOS_TRG);
     prev[0].push_back((unsigned int)SOS_TRG);
   }
   for (int t = 0; t < LENGTH_LIMIT; ++t) {
     unsigned int end_count = 0;
     for(unsigned int bi=0; bi < bsize; bi++){
       if(osents[bi][t] == EOS_TRG){
         end_count++;
       }
     }
     if(end_count == bsize) break;
     Expression i_r_t = encdec.Decoder(cg, prev[t]);
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

template <class Builder>
void train(){
  SOS_SRC = d_src.Convert("<s>");
  EOS_SRC = d_src.Convert("</s>");
  cerr << "Reading source language training text from " << PATH_TRAIN_SRC << "...\n";
  FreqCut(PATH_TRAIN_SRC, d_src, INPUT_VOCAB_SIZE);
  d_src.Freeze(); // no new word types allowed
  d_src.SetUnk("<unk>");
  UNK_SRC = d_src.Convert("<unk>");
  INPUT_VOCAB_SIZE = d_src.size();
  LoadCorpus(PATH_TRAIN_SRC, SOS_SRC, EOS_SRC, d_src, training_src);

  SOS_TRG = d_trg.Convert("<s>");
  EOS_TRG = d_trg.Convert("</s>");
  cerr << "Reading target language training text from " << PATH_TRAIN_TRG << "...\n";
  FreqCut(PATH_TRAIN_TRG, d_trg, OUTPUT_VOCAB_SIZE);
  d_trg.Freeze(); // no new word types allowed
  d_trg.SetUnk("<unk>");
  UNK_TRG = d_trg.Convert("<unk>");
  OUTPUT_VOCAB_SIZE = d_trg.size();
  LoadCorpus(PATH_TRAIN_TRG, SOS_TRG, EOS_TRG, d_trg, training_trg);
  cerr << "Writing source dictionary to " << PATH_DICT_SRC << "...\n";
  {
    ofstream out(PATH_DICT_SRC);
    boost::archive::text_oarchive oa(out);
    oa << d_src;
  }
  cerr << "Writing target dictionary to " << PATH_DICT_TRG << "...\n";
  {
    ofstream out(PATH_DICT_TRG);
    boost::archive::text_oarchive oa(out);
    oa << d_trg;
  }
  // for sorting
  for(unsigned int i=0; i < training_src.size(); i++){
    pair<vector<int>, vector<int> > p(training_src.at(i), training_trg.at(i));
    training.push_back(p);
  }
  // creating mini-batches
  CompareString comp;
  sort(training.begin(), training.end(), comp);
  for(size_t i = 0; i < training.size(); i += BATCH_SIZE){
    for(size_t j = 1; j < BATCH_SIZE; ++j){
      while(training.at(i+j).first.size() < training.at(i).first.size()){ // source padding
        training.at(i+j).first.push_back(EOS_SRC);
      }
      while(training.at(i+j).second.size() < training.at(i).second.size()){ // target padding
        training.at(i+j).second.push_back(EOS_TRG);
      }
    }
  }
  cerr << "Reading source development text from " << PATH_DEV_SRC << "...\n";
  LoadCorpus(PATH_DEV_SRC, SOS_SRC, EOS_SRC, d_src, dev_src);
  cerr << "Reading target development text from " << PATH_DEV_TRG << "...\n";
  LoadCorpus(PATH_DEV_TRG, SOS_TRG, EOS_TRG, d_trg, dev_trg);
  // for sorting
  for(unsigned int i=0; i < dev_src.size(); i++){
    ParaSent p(dev_src.at(i), dev_trg.at(i));
    dev.push_back(p);
  }
  // creating mini-batches
  sort(dev.begin(), dev.end(), comp);
  for(size_t i = 0; i < dev.size(); i += BATCH_SIZE){
    for(size_t j = 1; j < BATCH_SIZE; ++j){
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
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;
  
  Model model;
  EncoderDecoder<Builder> encdec(model);
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);
  sgd->clip_threshold *= BATCH_SIZE;

  vector<unsigned> order((training.size()+BATCH_SIZE-1)/BATCH_SIZE);
  for (unsigned i = 0; i < order.size(); ++i){
    order[i] = i * BATCH_SIZE;
  }

  vector<unsigned> dev_order((dev.size()+BATCH_SIZE-1)/BATCH_SIZE);
  for (unsigned i = 0; i < dev_order.size(); ++i){
    dev_order[i] = i * BATCH_SIZE;
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
      unsigned bsize = std::min((unsigned)training.size() - order[si], BATCH_SIZE); // Batch size
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
      sgd->update();
    }
    sgd->update_epoch();
    sgd->status();
    cerr << " E = " << (loss / order.size()) << " ppl=" << exp(loss / order.size()) << ' ';
    
#if 0
    lm.RandomSample();
#endif
    // show score on dev data?
    double dloss = 0;
    for(unsigned int dsi=0; dsi < dev_order.size(); dsi++){
      ComputationGraph cg;
      unsigned dev_bsize = std::min((unsigned)dev_order.size() - dev_order[dsi], BATCH_SIZE); // Batch size
      Batch isents, osents;
      SentList results;
      ToBatch(dev_order[dsi], dev_bsize, dev, isents, osents);
      greedy_decode<Builder>(isents, results, encdec, cg);
      for(unsigned int i = 0; i< results.size(); i++){
        dloss -= f_measure(dev.at(i + dev_order[dsi]).second, results.at(i)); // future work : replace to bleu
        cerr << "ref" << endl;
        print_sent(dev.at(i + dev_order[dsi]).second);
        cerr << "hyp" << endl;
        print_sent(results.at(i));
      }
    }
    if (dloss < best) {
      best = dloss;
      ofstream out(PATH_MODEL);
      boost::archive::text_oarchive oa(out);
      oa << model;
    }
    ++lines;
    cerr << "\n***DEV [epoch=" << lines << "] F = " << (0 - dloss / (double)dev.size()) << ' ';
  }
  delete sgd;

}

template <class Builder>
void test(){
  cerr << "Reading source dictionary from " << PATH_TEST_SRC << "...\n";
  {
    string fname = PATH_DICT_SRC;
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> d_src;
  }
  cerr << "Reading target dictionary from " << PATH_TEST_SRC << "...\n";
  {
    string fname = PATH_DICT_TRG;
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> d_trg;
  }
  cerr << "Reading source test text from " << PATH_TEST_SRC << "...\n";
  LoadCorpus(PATH_TEST_SRC, SOS_SRC, EOS_SRC, d_src, test_src);
  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  Model model;
  EncoderDecoder<Builder> encdec(model);
  string fname = PATH_MODEL;
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  // creating mini-batches
  CompareString comp;
  sort(test_src.begin(), test_src.end(), comp);
  for(size_t i = 0; i < test_src.size(); i += BATCH_SIZE){
    for(size_t j = 1; j < BATCH_SIZE; ++j){
      while(test_src.at(i+j).size() < test_src.at(i).size()){ // source padding
        test_src.at(i+j).push_back(EOS_SRC);
      }
    }
  }
  vector<unsigned> test_order((test_src.size()+BATCH_SIZE-1)/BATCH_SIZE);
  for (unsigned i = 0; i < test_order.size(); ++i){
    test_order[i] = i * BATCH_SIZE;
  }
  for(unsigned int tsi=0; tsi < test_order.size(); tsi++){
    ComputationGraph cg;
    unsigned test_bsize = std::min((unsigned)test_order.size() - test_order[tsi], BATCH_SIZE); // Batch size
    Batch isents, osents;
    SentList results;
    ToBatch(test_order[tsi], test_bsize, test_src, isents);
    greedy_decode<Builder>(isents, results, encdec, cg);
    for(unsigned int i = 0; i < results.size(); i++){
      print_sent(test_src.at(i + test_order[tsi]));
      print_sent(results.at(i));
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
  cnn::Initialize(argc, argv);
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
  ("batch-size",po::value<int>()->default_value(1), "batch size")
  ("beam-size", po::value<int>()->default_value(1), "beam size")
  ("src-vocab-size", po::value<int>()->default_value(20000), "source vocab size")
  ("trg-vocab-size", po::value<int>()->default_value(20000), "target vocab size")
  ("builder", po::value<int>()->default_value(0), "select builder (0:LSTM (default), 1:Fast-LSTM)")
  ("train", po::value<int>()->default_value(1), "is training ? (1:Yes,0:No)")
  ("test", po::value<int>()->default_value(1), "is test ? (1:Yes, 0:No)")
  ("dim-input", po::value<int>()->default_value(500), "dimmension size of embedding layer")
  ("dim-hidden", po::value<int>()->default_value(500), "dimmension size of hidden layer")
  ("layer-depth", po::value<int>()->default_value(1), "depth of hidden layer")
  ("cnn-mem", po::value<string>()->default_value("512m"), "memory size");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);
  INPUT_VOCAB_SIZE = vm.at("src-vocab-size").as<int>();
  OUTPUT_VOCAB_SIZE = vm.at("trg-vocab-size").as<int>();
  LAYERS = vm.at("layer-depth").as<int>();
  INPUT_DIM = vm.at("dim-input").as<int>();
  HIDDEN_DIM = vm.at("dim-hidden").as<int>();
  BATCH_SIZE = vm.at("batch-size").as<int>();
  BEAM_SIZE = vm.at("beam-size").as<int>();
  TRAIN = vm.at("train").as<int>();
  TEST = vm.at("test").as<int>();
  PATH_TRAIN_SRC = vm.at("path_train_src").as<string>();
  PATH_TRAIN_TRG = vm.at("path_train_trg").as<string>();
  PATH_DEV_SRC = vm.at("path_dev_src").as<string>();
  PATH_DEV_TRG = vm.at("path_dev_trg").as<string>();
  PATH_TEST_SRC = vm.at("path_test_src").as<string>();
  PATH_TEST_OUT = vm.at("path_test_out").as<string>();
  PATH_DICT_SRC = vm.at("path_dict_src").as<string>();
  PATH_DICT_TRG = vm.at("path_dict_trg").as<string>();
  PATH_MODEL = vm.at("path_model").as<string>();
  if(TRAIN > 0){
    switch(vm.at("builder").as<int>()){
      case __LSTM__:
      train<LSTMBuilder>();
      break;
      case __FAST_LSTM__:
      train<FastLSTMBuilder>();
      break;
    }
  }
  if(TEST > 0){
    switch(vm.at("builder").as<int>()){
      case __LSTM__:
      test<LSTMBuilder>();
      break;
      case __FAST_LSTM__:
      test<FastLSTMBuilder>();
      break;
    }
  }
}
