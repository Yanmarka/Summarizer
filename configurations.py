import trax

def c1_transformer(mode):
  return trax.models.TransformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode)

def c1_reformer(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode, ff_activation=trax.layers.Relu)

def c2_transformer(mode):
  return trax.models.TransformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=6, mode=mode)

def c2_reformer(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=6, mode=mode, ff_activation=trax.layers.Relu)

def c3_transformer(mode):
  return trax.models.TransformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=2048, n_layers=3, mode=mode)

def c3_reformer(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=2048, n_layers=3, mode=mode, ff_activation=trax.layers.Relu)

def c4_transformer(mode):
  return trax.models.TransformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=2048, n_layers=6, mode=mode)

def c4_reformer(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=2048, n_layers=6, mode=mode, ff_activation=trax.layers.Relu)

def lsh_reformer(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode, ff_activation=trax.layers.Relu, attention_type=trax.layers.LSHSelfAttention)