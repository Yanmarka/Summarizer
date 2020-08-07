import os
import summarizer
import trax

summarize_inputs = trax.supervised.inputs.Inputs(summarizer.lm_input_function)
my_trainer = summarizer.create_trainer(summarizer.my_transformer_lm, summarize_inputs, os.path.expanduser("~/output_dir"))
my_trainer.train_epoch(1,1)
