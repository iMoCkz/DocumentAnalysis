import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pathlib import Path


# root dir
root = 'C:/Users/Maxe/PycharmProjects/DocumentAnalysis/'
#
words_to_find = ['vehicle', 'automotive']
# tf_idf file writing
wrote_tf_idf_header = False
tf_idf_file_idx = 0
#
vectorizer_tf_idf = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None, vocabulary=words_to_find)
vectorizer_cnt = CountVectorizer(stop_words=None, vocabulary=words_to_find)
#
years = ['2018', '2019']
year_folders = [root + folder for folder in years]
# remove previous results file
if os.path.isfile('summary.csv'):
    os.remove('summary.csv')
if os.path.isfile('tf_idf.csv'):
    os.remove('tf_idf.csv')
#process every folder (for every year)
for year_idx, year_folder in enumerate(year_folders):
    # get file paths in folder
    file_paths = []
    for file in Path(year_folder).rglob("*.txt"):
        file_paths.append(file)
    # count of files for each year
    file_cnt = len(file_paths)
    # read every file's text as string
    docs_per_year = []
    words_in_folder = 0
    for txt_file in file_paths:
        with open(txt_file, encoding='utf-8', errors="replace") as f:
            txt_file_as_string = f.read()
            words_in_folder += len(txt_file_as_string.split())
            docs_per_year.append(txt_file_as_string)
    #
    tf_idf_documents_as_array = vectorizer_tf_idf.fit_transform(docs_per_year).toarray()
    # tf_idf_documents_as_array = vectorizer_tf_idf.fit_transform([' '.join(docs_per_year)]).toarray()
    #
    cnt_documents_as_array = vectorizer_cnt.fit_transform(docs_per_year).toarray()
    #
    with open('summary.csv', 'a') as f:
        f.write('Index;Term;Count;Df;Idf;Rel. Frequency\n')
        for idx, word in enumerate(words_to_find):
            abs_freq = cnt_documents_as_array[:, idx].sum()
            f.write('{};{};{};{};{};{}\n'.format(idx + 1,
                                                    word,
                                                    np.count_nonzero(cnt_documents_as_array[:, idx]),
                                                    abs_freq,
                                                    vectorizer_tf_idf.idf_[idx],
                                                    abs_freq / words_in_folder))
        f.write('\n')

    with open('tf_idf.csv', 'a') as f:
        if not wrote_tf_idf_header:
            f.write('{}\n'.format(years[year_idx]))
            f.write('Index;Year;File;')
            for word in words_to_find:
                f.write('{};'.format(word))
            f.write('Sum\n')
            wrote_tf_idf_header = True

        for idx, tf_idfs in enumerate(tf_idf_documents_as_array):
            f.write('{};{};{};'.format(tf_idf_file_idx, years[year_idx], file_paths[idx].name))
            for word_idx, _ in enumerate(words_to_find):
                f.write('{};'.format(tf_idf_documents_as_array[idx][word_idx]))
            f.write('{}\n'.format(sum(tf_idf_documents_as_array[idx])))

            tf_idf_file_idx += 1





