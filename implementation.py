import tensorflow as tf
import re

BATCH_SIZE = 32
MAX_WORDS_IN_REVIEW = 110  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'a', 'able', 'about', 'above', 'abst', 'accordance', 'according', 
    'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'affected', 'affecting', 
    'affects', 'after', 'afterwards', 'again', 'against', 'ah', 'all', 'almost', 'alone', 
    'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 
    'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 
    'anyway', 'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent', 
    'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available', 'away', 
    'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 
    'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 
    'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 
    'briefly', 'but', 'by', 'c', 'ca', 'came', 'can', 'cannot', "can't", 'cause', 'causes', 
    'certain', 'certainly', 'co', 'com', 'come', 'comes', 'contain', 'containing', 'contains', 
    'could', 'couldnt', 'd', 'date', 'did', "didn't", 'different', 'do', 'does', "doesn't", 
    'doing', 'done', "don't", 'down', 'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu', 
    'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 
    'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 
    'everything', 'everywhere', 'ex', 'except', 'f', 'far', 'few', 'ff', 'fifth', 'first', 
    'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 
    'found', 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 
    'give', 'given', 'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 
    'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', 'hed', 'hence', 
    'her', 'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 
    'hes', 'hi', 'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit', 'however', 
    'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im', 'immediate', 'immediately', 'importance', 
    'important', 'in', 'inc', 'indeed', 'index', 'information', 'instead', 'into', 'invention', 
    'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself', "i've", 'j', 'just', 'k', 
    'keep', 'keeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely', 'last', 
    'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 
    'liked', 'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks', 'ltd', 'm', 'made', 
    'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 
    'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most', 
    'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 
    'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 
    'never', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 
    'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'now', 'nowhere', 
    'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 
    'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others', 
    'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 
    'own', 'p', 'page', 'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 
    'placed', 'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 
    'predominantly', 'present', 'previously', 'primarily', 'probably', 'promptly', 'proud', 
    'provides', 'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 
    'readily', 'really', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 
    'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 
    'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'section', 'see', 'seeing', 
    'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven', 'several', 
    'shall', 'she', 'shed', "she'll", 'shes', 'should', "shouldn't", 'show', 'showed', 'shown', 
    'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly', 'since', 'six', 
    'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime', 
    'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 
    'specifying', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 
    'sufficiently', 'suggest', 'sup', 'sure', 't', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 
    'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their', 
    'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 
    'therefore', 'therein', "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 
    "there've", 'these', 'they', 'theyd', "they'll", 'theyre', "they've", 'think', 'this', 'those', 
    'thou', 'though', 'thoughh', 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 
    'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 
    'trying', 'ts', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 
    'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 
    'uses', 'using', 'usually', 'v', 'value', 'various', "'ve", 'very', 'via', 'viz', 'vol', 'vols', 
    'vs', 'w', 'want', 'wants', 'was', 'wasnt', 'way', 'we', 'wed', 'welcome', "we'll", 'went', 'were', 
    'werent', "we've", 'what', 'whatever', "what'll", 'whats', 'when', 'whence', 'whenever', 'where', 
    'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 
    'which', 'while', 'whim', 'whither', 'who', 'whod', 'whoever', 'whole', "who'll", 'whom', 
    'whomever', 'whos', 'whose', 'why', 'widely', 'willing', 'wish', 'with', 'within', 'without', 
    'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y', 'yes', 'yet', 'you', 'youd', 
    "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves', "you've", 'z', 'zero'})



def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # # lower all cases and replace breaklines to 1 space
    review = review.lower().replace('<br />', ' ')

    # replace some punctuations to 1 space
    review = re.sub(r'[.,\'\"]', ' ', review)

    # extend 1 space to 2 spaces for replacing stop-words
    review = review.replace(' ', '  ')

    # remove stop-words
    processed_review = re.sub(' '+' | '.join(list(stop_words))+' ', '', ' '+review+' ').split()

    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    numClasses = 2
    lstmUnits = 64
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape=())
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    learning_rate = 0.001

    labels = tf.placeholder(tf.float32, [BATCH_SIZE, numClasses], name="labels")
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)
    value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = tf.matmul(last, weight) + bias

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name="loss") # + regularization
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)   # learning_rate

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
