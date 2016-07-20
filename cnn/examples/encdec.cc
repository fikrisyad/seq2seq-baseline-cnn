#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
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
vector<vector<int>> training_src, training_trg, dev_src, dev_trg, test_src, test_out;
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
  void Encoder(const vector<int>& insent, ComputationGraph& cg) {
    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (unsigned t = 0; t < insent.size(); ++t) {
    	Expression i_x_t = lookup(cg,p_ec,insent[t]);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int t = insent.size() - 1; t >= 0; --t) {
      Expression i_x_t = lookup(cg, p_ec, insent[t]);
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
/*
    // decoder
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    vector<Expression> errs;

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
    	Expression i_x_t = lookup(cg, p_c, osent[t]);
    	Expression i_y_t = dec_builder.add_input(i_x_t);
    	Expression i_r_t = i_bias + i_R * i_y_t;
    	Expression i_err = pickneglogsoftmax(i_r_t,osent[t+1]);
    	errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
*/
  }

  Expression Decoder(ComputationGraph& cg, const int prev) {
    // decode
    Expression i_x_t = lookup(cg, p_c, prev);
    Expression i_y_t = dec_builder.add_input(i_x_t);
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);
    Expression i_r_t = i_bias + i_R * i_y_t;
    return i_r_t;
  }
};

void print_sent(vector<int>& osent){
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
double f_measure(vector<int> &isent, vector<int> &osent){
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

void greedy_decode(vector<int> &sent, vector<int> &osent, EncoderDecoder<LSTMBuilder> &encdec, ComputationGraph &cg){
   encdec.Encoder(sent, cg);
   osent.push_back(SOS_TRG);
   for (unsigned t = 0; t < LENGTH_LIMIT; ++t) {
     if(osent[t] == EOS_TRG) break;
     Expression i_r_t = encdec.Decoder(cg, osent[t]);
     Expression predict = softmax(i_r_t);
     auto output = as_vector(cg.incremental_forward());
     unsigned int w_id = 0;
     double w_prob = output[w_id];
     for(unsigned int i=0; i<output.size(); i++){
       double i_prob = output[i];
       if(i_prob > w_prob){
         w_id = i;
         w_prob = i_prob;
       }
     }
     osent.push_back(w_id);
   }
}

void train(){
  {
    SOS_SRC = d_src.Convert("<s>");
    EOS_SRC = d_src.Convert("</s>");
    cerr << "Reading source language training text from " << PATH_TRAIN_SRC << "...\n";
    ifstream in(PATH_TRAIN_SRC);
    assert(in);
    int tlc = 0;
    int ttoks = 0;
    string line;
    while(getline(in, line)) {
      ++tlc;
      training_src.push_back(ReadSentence(line, &d_src));
      ttoks += training_src.back().size();
      if (training_src.back().front() != SOS_SRC && training_src.back().back() != EOS_SRC) {
	cerr << "Training sentence in " << PATH_TRAIN_SRC << ":" << tlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d_src.size() << " types\n";
    d_src.Freeze(); // no new word types allowed
    d_src.SetUnk("<unk>");
    UNK_SRC = d_src.Convert("<unk>");
    INPUT_VOCAB_SIZE = d_src.size();
  }
  {
    SOS_TRG = d_trg.Convert("<s>");
    EOS_TRG = d_trg.Convert("</s>");
    cerr << "Reading target language training text from " << PATH_TRAIN_TRG << "...\n";
    ifstream in(PATH_TRAIN_TRG);
    assert(in);
    int tlc = 0;
    int ttoks = 0;
    string line;
    while(getline(in, line)) {
      ++tlc;
      training_trg.push_back(ReadSentence(line, &d_trg));
      ttoks += training_trg.back().size();
      if (training_trg.back().front() != SOS_TRG && training_trg.back().back() != EOS_TRG) {
	cerr << "Training sentence in " << PATH_TRAIN_TRG << ":" << tlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d_trg.size() << " types\n";
    d_trg.Freeze(); // no new word types allowed
    d_trg.SetUnk("<unk>");
    UNK_TRG = d_trg.Convert("<unk>");
    OUTPUT_VOCAB_SIZE = d_trg.size();
  }
  {
    cerr << "Reading source development text from " << PATH_DEV_SRC << "...\n";
    int dlc = 0;
    int dtoks = 0;
    ifstream in(PATH_DEV_SRC);
    assert(in);
    string line;
    while(getline(in, line)) {
      ++dlc;
      dev_src.push_back(ReadSentence(line, &d_src));
      dtoks += dev_src.back().size();
      if (dev_src.back().front() != SOS_SRC && dev_src.back().back() != EOS_SRC) {
	cerr << "Dev sentence in " << PATH_DEV_SRC << ":" << dlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  {
    cerr << "Reading target development text from " << PATH_DEV_TRG << "...\n";
    int dlc = 0;
    int dtoks = 0;
    ifstream in(PATH_DEV_TRG);
    assert(in);
    string line;
    while(getline(in, line)) {
      ++dlc;
      dev_trg.push_back(ReadSentence(line, &d_trg));
      dtoks += dev_trg.back().size();
      if (dev_trg.back().front() != SOS_TRG && dev_trg.back().back() != EOS_TRG) {
	cerr << "Dev sentence in " << PATH_DEV_TRG << ":" << dlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
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
  EncoderDecoder<LSTMBuilder> encdec(model);
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);
  
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = 10;
  unsigned si = training_src.size();
  vector<unsigned> order(training_src.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == training_src.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      // build graph for this instance
      ComputationGraph cg;
      auto& sent = training_src[order[si]];
      auto& osent = training_trg[order[si]];
      chars += sent.size() - 1;
      ++si;
      encdec.Encoder(sent, cg);
      vector<Expression> errs;
      const unsigned oslen = osent.size() - 1;
      for (unsigned t = 0; t < oslen; ++t) {
        Expression i_r_t = encdec.Decoder(cg, osent[t]);
        Expression i_err = pickneglogsoftmax(i_r_t,osent[t+1]);
        errs.push_back(i_err);
      }
      Expression i_nerr = sum(errs);
      //cg.PrintGraphviz();
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
    
#if 0
    lm.RandomSample();
#endif
    
    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for(unsigned int i=0; i < dev_src.size(); i++){
	ComputationGraph cg;
        vector<int> osent;
	greedy_decode(dev_src.at(i), osent, encdec, cg);
	dloss -= f_measure(dev_src.at(i), osent); // future work : replace to bleu
	dchars += osent.size() - 1;
        print_sent(dev_src.at(i));
        print_sent(osent);
      }
      if (dloss < best) {
	best = dloss;
	ofstream out(PATH_MODEL);
	boost::archive::text_oarchive oa(out);
	oa << model;
      }
      cerr << "\n***DEV [epoch=" << (lines / (double)training_src.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
    }
  }
  delete sgd;

}


void test(){
  {
    cerr << "Reading source test text from " << PATH_TEST_SRC << "...\n";
    int tlc = 0;
    int ttoks = 0;
    ifstream in(PATH_TEST_SRC);
    assert(in);
    string line;
    while(getline(in, line)) {
      ++tlc;
      test_src.push_back(ReadSentence(line, &d_src));
      ttoks += test_src.back().size();
      if (test_src.back().front() != SOS_SRC && test_src.back().back() != EOS_SRC) {
	cerr << "Test sentence in " << PATH_TEST_SRC << ":" << tlc << " didn't start or end with <s>, </s>\n";
	abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens\n";
  }
  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  Model model;
  EncoderDecoder<LSTMBuilder> encdec(model);
  string fname = PATH_MODEL;
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  for(unsigned int i=0; i < test_src.size(); i++){
    ComputationGraph cg;
    vector<int> osent;
    greedy_decode(test_src.at(i), osent, encdec, cg);
    print_sent(osent);
    test_out.push_back(osent);
  }

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
  ("train", po::value<int>()->default_value(1), "is training ? (1:Yes,0:No)")
  ("test", po::value<int>()->default_value(1), "is test ? (1:Yes, 0:No)")
  ("dim-input", po::value<int>()->default_value(500), "dimmension size of embedding layer")
  ("dim-hidden", po::value<int>()->default_value(500), "dimmension size of hidden layer")
  ("size-layer", po::value<int>()->default_value(1), "layer size of hidden layer");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);
  LAYERS = vm.at("size-layer").as<int>();
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
   train();
  }
  if(TEST > 0){
   test();
  }
}
