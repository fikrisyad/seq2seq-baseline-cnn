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

#include "Cho2014.hpp"
#include "Sutskever2014.hpp"
#include "Bahdanau2015.hpp"
#include "encdec.hpp"
#include "decode.hpp"
#include "define.hpp"
#include "comp.hpp"
#include "preprocess.hpp"
#include "metrics.hpp"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

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
  vm.at("src-vocab-size").value() = d_src.size();
  //vm.at("src-vocab-size").as<int>() = d_src.size();
  LoadCorpus(vm.at("path_train_src").as<string>(), SOS_SRC, EOS_SRC, d_src, training_src);

  SOS_TRG = d_trg.Convert("<s>");
  EOS_TRG = d_trg.Convert("</s>");
  cerr << "Reading target language training text from " << vm.at("path_train_trg").as<string>() << "...\n";
  FreqCut(vm.at("path_train_trg").as<string>(), d_trg, vm.at("trg-vocab-size").as<unsigned int>());
  d_trg.Freeze(); // no new word types allowed
  d_trg.SetUnk("<unk>");
  UNK_TRG = d_trg.Convert("<unk>");
  vm.at("trg-vocab-size").value() = d_trg.size();
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
  EncoderDecoder<Builder>* encdec;
  //AttentionalEncoderDecoder<Builder> encdec(model, vm);
  switch(vm.at("encdec-type").as<unsigned int>()){
    case __Cho2014__:
    encdec = new Cho2014<Builder>(model, &vm);
    break;
    case __Sutskever2014__:
    encdec = new Sutskever2014<Builder>(model, &vm);
    break;
    case __Bahdanau2015__:
    encdec = new Bahdanau2015<Builder>(model, &vm);
    break;
  }
  Trainer* trainer = nullptr;
  switch(vm.at("trainer").as<unsigned int>()){
    case __SGD__:
    trainer = new SimpleSGDTrainer(&model);
    break;
    case __MomentumSGD__:
    trainer = new MomentumSGDTrainer(&model);
    break;
    case __Adagrad__:
    trainer = new AdagradTrainer(&model);
    break;
    case __Adadelta__:
    trainer = new AdadeltaTrainer(&model);
    break;
    case __RMSprop__:
    trainer = new RmsPropTrainer(&model);
    break;
    case __Adam__:
    trainer = new AdamTrainer(&model);
    break;
  }
  trainer->eta = vm.at("eta").as<float>();
  trainer->clip_threshold *= vm.at("batch-size").as<unsigned int>();
  // Set the start point for each mini-batch of training dataset 
  vector<unsigned> order((training.size()+vm.at("batch-size").as<unsigned int>()-1)/vm.at("batch-size").as<unsigned int>());
  for (unsigned i = 0; i < order.size(); ++i){
    order[i] = i * vm.at("batch-size").as<unsigned int>();
  }

  // Set the start point for each mini-batch of development dataset 
  vector<unsigned> dev_order((dev.size()+vm.at("parallel").as<unsigned int>()-1)/vm.at("parallel").as<unsigned int>());
  for (unsigned i = 0; i < dev_order.size(); ++i){
    dev_order[i] = i * vm.at("parallel").as<unsigned int>();
  }

  unsigned lines = 0;
  while(1) {
    cerr << "**SHUFFLE\n";
    shuffle(order.begin(), order.end(), *rndeng);
    Timer iteration("completed in");
    double loss = 0;
    for (unsigned si = 0; si < order.size(); ++si) {
      // build graph for this instance
      unsigned bsize = std::min((unsigned)training.size() - order[si], vm.at("batch-size").as<unsigned int>()); // Batch size
      for(int i = 0, offset = order[si]; i <= bsize / vm.at("parallel").as<unsigned int>(); i++){
        unsigned split_size = 0;
        split_size = std::min(order[si] + bsize - offset, vm.at("parallel").as<unsigned int>());
        if(split_size == 0) break;
        ComputationGraph cg;
        vector<Expression> errs;
        Batch sents, osents;
        ToBatch(offset, split_size, training, sents, osents);
        encdec->Encoder(sents, cg);
        {
          Expression i_r_t = encdec->Decoder(cg);
          Expression i_err = pickneglogsoftmax(i_r_t, osents[0]);
        }
        for (int t = 0; t < osents.size() - 1; ++t) {
          Expression i_r_t = encdec->Decoder(cg, osents[t]);
          //vector<unsigned int> next = osents[t+1];
          Expression i_err = pickneglogsoftmax(i_r_t, osents[t+1]);
          errs.push_back(i_err);
        }
        Expression i_nerr = sum_batches(sum(errs));
        //cg.PrintGraphviz();
        loss += as_scalar(cg.forward()) / (double)bsize;
        cg.backward();
        offset += split_size;
      }
      trainer->update((1.0 / double(bsize)));
      //sgd->update();
      cerr << " E = " << (loss / double(si + 1)) << " ppl=" << exp(loss / double(si + 1)) << ' ';
      cerr  << "source length=" << training.at(order[si]).first.size() << " target length=" << training.at(order[si]).second.size() << std::endl;
    }
    trainer->update_epoch();
    trainer->status();
    
#if 0
    lm.RandomSample();
#endif
    // show score on dev data?
    double dloss = 0;
    for(unsigned int dsi=0; dsi < dev_order.size(); dsi++){
      ComputationGraph cg;
      unsigned dev_bsize = std::min((unsigned)dev.size() - dev_order[dsi], vm.at("parallel").as<unsigned int>()); // Batch size
      Batch isents, osents;
      SentList results;
      ToBatch(dev_order[dsi], dev_bsize, dev, isents, osents);
      Decode::Greedy<Builder>(isents, results, encdec, cg, vm);
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
  delete trainer;

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
  EncoderDecoder<Builder>* encdec;
  switch(vm.at("encdec-type").as<unsigned int>()){
    case __Cho2014__:
    encdec = new Cho2014<Builder>(model, &vm);
    break;
    case __Sutskever2014__:
    encdec = new Sutskever2014<Builder>(model, &vm);
    break;
    case __Bahdanau2015__:
    encdec = new Bahdanau2015<Builder>(model, &vm);
    break;
  }
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
    Decode::Greedy<Builder>(isents, results, encdec, cg, vm);
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
  ("parallel",po::value<unsigned int>()->default_value(1), "parallel size")
  ("beam-size", po::value<unsigned int>()->default_value(1), "beam size")
  ("src-vocab-size", po::value<unsigned int>()->default_value(20000), "source vocab size")
  ("trg-vocab-size", po::value<unsigned int>()->default_value(20000), "target vocab size")
  ("builder", po::value<unsigned int>()->default_value(0), "select builder (0:LSTM (default), 1:Fast-LSTM, 2:GRU, 3:RNN)")
  ("trainer", po::value<unsigned int>()->default_value(0), "select trainer (0:SGD (default), 1:MomentumSGD, 2:Adagrad, 3:Adadelta, 4:RMSprop, 5:Adam)")
  ("encdec-type", po::value<unsigned int>()->default_value(2), "select a type of encoder-decoder (0:cnn example, 1:encoder-decoder, 2:attention (default))")
  ("train", po::value<unsigned int>()->default_value(1), "is training ? (1:Yes,0:No)")
  ("test", po::value<unsigned int>()->default_value(1), "is test ? (1:Yes, 0:No)")
  ("dim-input", po::value<unsigned int>()->default_value(500), "dimmension size of embedding layer")
  ("dim-hidden", po::value<unsigned int>()->default_value(500), "dimmension size of hidden layer")
  ("dim-attention", po::value<unsigned int>()->default_value(64), "dimmension size of hidden layer")
  ("depth-layer", po::value<unsigned int>()->default_value(1), "depth of hidden layer")
  ("limit-length", po::value<unsigned int>()->default_value(100), "length limit of target language in decoding")
  ("eta", po::value<float>()->default_value(1.0), "learning rate")
  ("cnn-mem", po::value<string>()->default_value("512m"), "memory size");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);
  cnn::Initialize(argc, argv);
  if(vm.at("train").as<unsigned int>() > 0){
    switch(vm.at("builder").as<unsigned int>()){
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
  if(vm.at("test").as<unsigned int>() > 0){
    switch(vm.at("builder").as<unsigned int>()){
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
