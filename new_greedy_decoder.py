import numpy as np

# This code is based on decoding code provided by Lukasz Kaiser
def next_symbol(cur_output_tokens, model):
  #padded_length = min([2**int(np.ceil(np.log2(len(cur_output_tokens) + 3))), 2048])
  padded_length = 4096
  padding_difference = padded_length - len(cur_output_tokens)
  if padding_difference >= 0:
    padded = cur_output_tokens + [0] * padding_difference
  else:
    padding_difference = padding_difference * (-1)
    padded = cur_output_tokens[padding_difference:]
  padded_with_batch = np.array(padded)[None, :]
  output, _ = model((padded_with_batch, padded_with_batch), n_accelerators=1)
  log_probs = output[0, len(cur_output_tokens), :]
  print(int(np.argmax(log_probs)))
  return int(np.argmax(log_probs))

def greedy_predict(article, model, summary_limit=400):
  cur_output_tokens = article + [1]
  generated_output = []
  cur_output = 0
  EOS = 1
  i = 0
  while cur_output != EOS and i <= 10:
    cur_output = next_symbol(cur_output_tokens, model)
    cur_output_tokens.append(cur_output)
    generated_output.append(cur_output)
    #print(tokenizer.DecodeIds(generated_output))
    i += 1
  return generated_output