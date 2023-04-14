# this code aims to convert the conllu-formatted data to csv and divide it to 3 datasets
# the converter of the conllu-formatted data to the csv format is based on the CoNLL-U_Parser algorythm (https://github.com/MuhammadYaseenKhan/CoNLL-U_Parser)

import pandas as pd
from time import time
from sklearn.utils import shuffle

# implementation of the CoNLL-U_Parser
def convert_data(file_name):
    start = time()
    file_path = f'{file_name}.conllu'  # path where your .conllu file is located
    with open(file_path, 'r', encoding='utf-8') as file:
        file_prefix = file_path.split('.')[0] + '_'
        doc_id = ''
        sent_id = ''
        records = list()
        for line in file:
            if len(line) > 1:
                if line[0] == '#':
                    line = line.split('=')
                    if 'newdoc' in line[0]:
                        doc_id = file_prefix + line[1].strip()
                    elif 'sent_id' in line[0]:
                        sent_id = line[1].strip()
                else:
                    info = line.split('\t')
                    if len(info) == 10:
                        records.append([doc_id, sent_id] + [x.strip() for x in info])
    end = time()
    print("Time elapsed:", end - start, "seconds")

    df = pd.DataFrame(records,
                      columns=['DOC_NO', 'SENT_NO', 'ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEAT', 'HEAD', 'DEPREL',
                               'DEPS', 'MISC'])
    print(df.head())



    df.to_csv(f'{file_name}.tsv', header=True, index=False, sep='\t')
    df_word = df['FORM'].apply(str. lower).str.replace('[^\w\']','').drop_duplicates()
    df_word = df_word[df_word != ""]
    df_word.to_csv(f'{file_name}.txt', sep=' ', header=False, index=False)
    split_files(df_word, file_name)

# split the data with the ratio 60-20-20
def split_files(file, prefix):
    # Split the data into train, dev, and test sets
    # shuffle the DataFrame randomly
    data = shuffle(file)
    # split the DataFrame into train, dev, and test sets
    train = data[:int(0.6 * len(data))]
    dev = data[int(0.6 * len(data)):int(0.8 * len(data))]
    test = data[int(0.8 * len(data)):]
    print("Size of train set:", len(train))
    print("Size of dev set:", len(dev))
    print("Size of test set:", len(test))
    # train.to_csv(f'{prefix}_train.txt', sep=' ', header=False, index=False)
    # test.to_csv(f'{prefix}_test.txt', sep=' ', header=False, index=False)
    # dev.to_csv(f'{prefix}_dev.txt', sep=' ', header=False, index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   convert_data('xav')

