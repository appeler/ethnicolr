# -*- coding: utf-8 -*-

import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

def isstring(s):
    # if we use Python 3
    if (sys.version_info[0] >= 3):
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def column_exists(df, col):
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        print("The specify column `{0!s}` not found in the input file"
              .format(col))
        return False
    else:
        return True


def fixup_columns(cols):
    """Replace index location column to name with `col` prefix

    Args:
        cols (list): List of original columns

    Returns:
        list: List of column names

    """
    out_cols = []
    for col in cols:
        if type(col) == int:
            out_cols.append('col{:d}'.format(col))
        else:
            out_cols.append(col)
    return out_cols


def find_ngrams(vocab, text, n):
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    if not isstring(text):
        return wi

    a = zip(*[text[i:] for i in range(n)])
    for i in a:
        w = ''.join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi

def transform_and_pred(df = df, 
                       newnamecol, 
                       cls, 
                       VOCAB,
                       RACE,
                       MODEL,
                       NGRAMS,
                       maxlen, 
                       num_iter, 
                       conf_int):

    df[newnamecol] = df[newnamecol].str.strip().str.title()

    if cls.model is None:
        vdf = pd.read_csv(VOCAB)
        cls.vocab = vdf.vocab.tolist()

        rdf = pd.read_csv(RACE)
        cls.race = rdf.race.tolist()

        cls.model = load_model(MODEL)
    
    # build X from index of n-gram sequence
    X = np.array(df[newnamecol].apply(lambda c:
                                                 find_ngrams(cls.vocab,
                                                             c, NGRAMS)))
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # define the quantile ranges for the confidence interval
    low_quantile = 0.5 - (conf_int / 2)
    high_quantile = 0.5 + (conf_int / 2)
    
    # Predict
    proba = []
    for _ in range(num_iter):
        proba.append(cls.model.predict(X, verbose=2))

    # creating arrays for the confidence interval analysis:
    #   1 - mean of the predictions
    #   2 - std deviation of the predictions
    #   3 - lower quantile of the predictions
    #   4 - upper quantile of the predictions
    proba = np.array(proba)
    mean_arr = proba.mean(axis=0).reshape(-1, len(cls.race))
    std_arr = proba.std(axis=0).reshape(-1, len(cls.race))
    pct_low_arr = np.quantile(proba, low_quantile, axis=0).reshape(-1, len(cls.race))
    pct_high_arr = np.quantile(proba, high_quantile, axis=0).reshape(-1, len(cls.race))

    df.loc[,'__pred'] = np.argmax(mean_arr, axis=-1)

    df.loc[,'race'] = df['__pred'].apply(lambda c:
                                                    cls.race[int(c)])
    stats = np.zeros((df.shape[0], 4))
    conf_int = []

    # selecting the statistics of the chosen class
    for i in range(df.shape[0]):
        select_class = np.argmax(mean_arr[i], axis=-1)
        stats[i, 0] = mean_arr[i, select_class]
        stats[i, 1] = std_arr[i, select_class]
        stats[i, 2] = pct_low_arr[i, select_class]
        stats[i, 3] = pct_high_arr[i, select_class]
        conf_int.append(np.array([stats[i, 2], stats[i, 3]]).tolist())

        df['proba'] = stats[:, 0]
        df['std_err'] = stats[:, 1]
        df['conf_int'] = conf_int
    
    # take out temporary working columns
    del df['__pred']
    del df[newnamecol]

    pdf = pd.DataFrame(proba, columns=cls.race)
    pdf.set_index(df.index, inplace=True)

    rdf = pd.concat([df, pdf], axis=1)

    return rdf
