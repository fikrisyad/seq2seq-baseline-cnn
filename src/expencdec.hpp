
template <class Builder>
struct ExampleEncoderDecoder {
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

