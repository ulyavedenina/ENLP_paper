# this is the implementation of the code based on the description from the following github page: https://github.com/rnd2110/MorphAGram
# the code below analyses the performance of the unsupervised model based on Teko data
# the preprocessing of the data set for Teko can be found in retrieve_data.py
from sklearn.metrics import f1_score
from analysis import analyze_output, analyze_gold
from preprocessing import process_words, write_encoded_words, read_grammar, add_chars_to_grammar, write_grammar
from segmentation import generate_grammar, parse_segmentation_output, segment_file

# ---preprocessing---
# the path to the unsegmented train+dev data set
lexicon_path = 'data/Teko/data/eme_traindev.txt'
# the path to the Hex-encoded train+dev data set
encoded_lexicon_path = 'data/Teko/data/lexicon.txt'
# the path to SFG used (9 in total)
grammar_number = 19
grammar_path = f'data/Teko/grammar/standard/grammar{grammar_number}.txt'
# the path to the SFG used + appended Hex-encoded chars as terminals
final_grammar_path = f'data/Teko/grammar/standard/final_grammar{grammar_number}.txt'

# Read the initial lexicon (word list), and convert it into Hex.
words, encoded_words, hex_chars = process_words(lexicon_path)
write_encoded_words(encoded_words, encoded_lexicon_path)

# Read the initial CFG and append the HEX encoded characters as terminals.
# encoded_lexicon_path and final_grammar_path then become the input to the PYAGS sampler.
grammar = read_grammar(grammar_path)
appended_grammar = add_chars_to_grammar(grammar, hex_chars)
write_grammar(appended_grammar, final_grammar_path)
#
# # ---training---
# # After that, the Pitman-Yor Adaptor-Grammar Sampler (PYAGS) was run (on AWS Cloud)
# # The results of the runs are located in data/Teko/pycfg/standard/output
#
# # ---segmentation---
# # If a word is seen in the training data, its segmentation
# # is read from the segmentation output of the learning process.
# # If a word is not seen in the training data, it's analyzed
# # by finding the split that gives the highest MLE probability
# # across its morphemes, along with the selection of compatible
# # prefixes and suffixes. The information of the morphemes and their
# # MLE probabilities and compatibility are driven from the segmentation
# # output of the learning process. # This method is applicable only when
# # the prefixes, stems and suffixes are represented by three different
# # nonterminals.
#
# # Create a segmentation model given the PYAGS segmentation output.
# # The step requires specifying which nonterminals to split on.
# # In addition to generating the segmentation model, the step generates
# # a human-readable segmentation output that can be directly used as the
# # prediction input for the evaluation scripts used in the Morpho-Challenge
# # shared task.
# # The parameter is optional. If used, it should be the ISO-639-1 language
# # code, and it only affects Turkish (for its special lowercasing and uppercasing).
#
# # path to the produced output during the PYAGS training
# pyags_output_grammar_path = '/Users/ulyavedenina/Documents/MorphAGram/data/Teko/pycfg/standard/output/grammar-traindev.cfg1-s.txt'
#
# segmentation_model = parse_segmentation_output(pyags_output_grammar_path,
#                                                'Prefix', 'Stem', 'Suffix', '/Users/ulyavedenina/Documents/MorphAGram/data/Teko/pycfg/standard/output/out.txt',
#                                                 None, 1)
#
# # segment validation and test sets. Validation test -- was seen during the training (train+dev); test set -- unseen
# segment_file('data/Teko/data/eme_dev_unseg.txt', 'data/Teko/data/eme_dev_res.txt', segmentation_model, ' ', ' ', False, None , 1)
# segment_file('data/Teko/data/eme_test_unseg.txt', 'data/Teko/data/eme_test_res.txt', segmentation_model, ' ', ' ', False, None , 1)
#
# # data set and output statistics
# gold_path = 'data/Teko/data/eme_dev.txt'
# output_path = 'data/Teko/pycfg/standard/output/out.txt'
# gold_info, morph_info = analyze_gold(gold_path)
# morph_out_info = analyze_output(output_path, gold_path)
#
# # ---evaluation---
# # evaluation for BPR and Emma-2 was performed with morphoeval library using the following script:
# # morphoeval [--metric] gold_values_file predicted_value_file output_file
# # the results are stored in /Users/ulyavedenina/Documents/MorphAGram/data/Teko/data
