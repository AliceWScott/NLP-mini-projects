import loader
from nltk import pos_tag
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords
from nltk import wsd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
from math import log
from itertools import product
from nltk.probability import FreqDist

'''
	Returns dictionary containing the most frequent sense baseline.
'''
def baseline(instances):

    results = []
    for key in instances.iterkeys():
    	lemma = instances[key].lemma
    	ss = wn.synsets(lemma)[0]
    	most_freq = ss.lemmas()[0]
    	results.append((key, most_freq))
    return results

'''
    Used to convert the returned POS tags from NLTK's pos_tag()
    to the proper input tags for the Lesk method.
'''
def convertPOS(tag):

    tag = tag[0][1]

    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

'''
    First part is based off NLTK Lesk Algorithm source code.
    Used for the combined() method.
'''
def probability_lesk(context, word, documents, pos=None, synsets=None):

    ss_frequencies = {}

    # count occurrences of each sense in WordNet.
    for ss in wn.synsets(word):
        frequencies = 0
        for l in ss.lemmas():
            frequencies += l.count()
        ss_frequencies[ss] = frequencies

    context = set(context)
    best_sense = None
    max_overlap = 0

    #creates a dictionary with the probabilities of each synset occuring in the corpus used by WordNet.
    total_count = sum([f[1] for f in ss_frequencies.items()])
    if total_count > 0:
        ss_probs = dict([(f[0], f[1]/float(total_count)) for f in ss_frequencies.items()])
    else:
        ss_probs = dict([(f[0], 0.0) for f in ss_frequencies.items()])

    if synsets is None:
        synsets = wn.synsets(word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None


    '''
    Instead of returning the sense with the highest count of intersecting words,
     we return the sense with the highest sum of IDFs in the intersection.
     This sum is then scaled to the overall probability of a sense occurring in the corpus used by WordNet.
    '''
    num_docs = len(documents)
    senses = []
    for ss in synsets:
        idf = []
        for w in context.intersection(ss.definition().split()):
            occurences = count_occurrences(w, documents)
            idf.append(log(num_docs / float(occurences)))

        freq = ss_probs[ss]
        senses.append((sum(idf)*freq, ss)) 

    _, sense = max(senses)
    return sense

'''
    First part is based off NLTK Lesk Algorithm source code.
    Used for the combined() method.
'''
def IDF_lesk(context, word, documents, pos=None, synsets=None):

    context = set(context)
    best_sense = None
    max_overlap = 0

    if synsets is None:
        synsets = wn.synsets(word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    '''
    Instead of returning the sense with the highest count of intersecting words,
     we return the sense with the highest sum of IDFs in the intersection.
    '''
    num_docs = len(documents)
    senses = []
    for ss in synsets:
        idf = []
        for w in context.intersection(ss.definition().split()):
            occurences = count_occurrences(w, documents)
            idf.append(log(num_docs / float(occurences)))
        senses.append((sum(idf), ss)) 
    _, sense = max(senses)
    return sense

'''
    Relies on the highest sum of Inverse Document Frequencies of words
    in the intersection of definitions and contexts.
'''
def combined(instances, n=None, useProbs=True, lemmatize=False, removeSW=False):

    lem = WordNetLemmatizer()
    sw = list(stopwords.words('english'))

    # create a list of all unique context sentences.
    all_contexts = set()
    for key in instances.iterkeys():
        sentence = [w.replace('_', ' ').translate(None, string.punctuation).replace(' ', '_') 
                    for w in instances[key].context]
        sentence = ' '.join(sentence)
        all_contexts.add(sentence)
    all_contexts = [s.split(' ') for s in all_contexts]

   
    results = []

    for key in instances.iterkeys():

    	curr_context = instances[key].context
    	ambiguous = instances[key].lemma
        tag = pos_tag([ambiguous])
        tag = convertPOS(tag)

    	context = []

    	# we need to create a new context with lemmatization
    	for word in curr_context:

    		# temporarily convert underscores to whitespace, remove punctuation
    	   word = word.replace('_', ' ').translate(None, string.punctuation).replace(' ', '_')
           if word: 
                if lemmatize and removeSW:
                     if word not in sw: 
                        context.append(lem.lemmatize(word))
                elif lemmatize:
                    context.append(lem.lemmatize(word))
                elif removeSW:
                    if word not in sw:
                        context.append(word)
                else:
                    context.append(word)
    		

        i = instances[key].index
        if n: context = context[i-n:i+n]

        # add the pos tag if there is one
        # Use the right modified Lesk algorithm, depending on if useProbs = True or not.
        if useProbs:
    	    top = probability_lesk(context, ambiguous, all_contexts, pos=tag)
            if top: 
                top = top.lemmas()[0]
            else:
                top = probability_lesk(context, ambiguous, all_contexts).lemmas()[0]
        else:
            top = IDF_lesk(context, ambiguous, all_contexts, pos=tag)
            if top:
                top = top.lemmas()[0]
            else:
                top = IDF_lesk(context, ambiguous, all_contexts).lemmas()[0]
    	results.append((key, top)) #select the sense with the highest overlap score

    return results


 

'''
    Helper function for modified Lesk algorithm. 
    Return the number of documents containing a word.
'''
def count_occurrences(word, documents):

    count = 0
    for d in documents:
        if word in d: count += 1

    return count


'''
	Based on the following paper:
	http://www.d.umn.edu/~tpederse/Pubs/cicling2002-b.pdf
	The context of a word is defined as a window of n WordNet word tokens to the left,
	and n to the right. The word is also included in the context, 
	so context size is a total of 2n+1 words.
'''
def lesk_alg(instances, n=None, lemmatize=True, removeSW=True):
    
    sw = list(stopwords.words('english'))
    lem = WordNetLemmatizer()

    results = []


    for key in instances.iterkeys():

        curr_context = instances[key].context
        ambiguous = instances[key].lemma
        tag = pos_tag([ambiguous])
        tag = convertPOS(tag)

        context = []

        # we need to create a new context with lemmatization and no stop words
        for word in curr_context:
            # convert underscores to whitespace, remove punctuation
            word = word.replace('_', ' ').translate(None, string.punctuation).replace(' ', '_')
            if word: 
                if lemmatize and removeSW:
                     if word not in sw: 
                        context.append(lem.lemmatize(word))
                elif lemmatize:
                    context.append(lem.lemmatize(word))
                elif removeSW:
                    if word not in sw:
                        context.append(word)
                else:
                    context.append(word)

        i = instances[key].index
        if n: context = context[i-n:i+n]
        top = wsd.lesk(context, ambiguous, pos=tag)
        if top: 
            top = top.lemmas()[0]
        else: 
            top = wsd.lesk(context, ambiguous).lemmas()[0]
        results.append((key, top)) #select the sense with the highest overlap score
    return results
	    

def getAccuracy(predicted, actual):
    count = 0
    for (key, result) in predicted:
        if result.key() in actual[key]: #in case there is more than one sense correct in key
            count += 1
    return float(count) / len(actual) * 100.0



if __name__ == '__main__':

    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = loader.load_instances(data_f)
    dev_key, test_key = loader.load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}


    lem = [True, False]
    removeSW = [True, False]
    window = range(1, 5) + [None]
    params = list(product(*[lem, removeSW, window]))

    base = baseline(dev_instances)
    print 'DEV BASELINE ACCURACY:', getAccuracy(base, dev_key)
    print '__________________'

    best_idf_p = (0, None)
    best_lesk_p = (0, None)
    best_prob_p = (0, None)

    for p in params:

        lesk = lesk_alg(dev_instances, lemmatize=p[0], removeSW=p[1], n=p[2])
        idf = combined(dev_instances, useProbs=False, lemmatize=p[0], removeSW=p[1], n=p[2])
        probs = combined(dev_instances, useProbs=True, lemmatize=p[0], removeSW=p[1], n=p[2])

        lesk_acc =  getAccuracy(lesk, dev_key)
        idf_acc = getAccuracy(idf, dev_key)
        probs_acc = getAccuracy(probs, dev_key)

        print 'CURRENT PARAMETERS:', p
        print 'DEV LESK ACCURACY:', lesk_acc
        print 'DEV IDF LESK ACCURACY:', idf_acc
        print 'DEV PROBABILITY LESK ACCURACY:', probs_acc
        print '__________________'

        if lesk_acc > best_lesk_p[0]: best_lesk_p = (lesk_acc, p)
        if idf_acc > best_idf_p[0]: best_idf_p = (idf_acc, p)
        if probs_acc > best_prob_p[0]: best_prob_p = (probs_acc, p)

    print 'BEST PARAMETERS FOR LESK ALGORITHM ARE:'
    print 'lemmatize=', best_lesk_p[1][0], ',remove stopwords=', best_lesk_p[1][1], ',window size=', best_lesk_p[1][2]
    print 'BEST PARAMETERS FOR IDF LESK ALGORITHM ARE:'
    print 'lemmatize=', best_idf_p[1][0], ',remove stopwords=', best_idf_p[1][1], ',window size=', best_idf_p[1][2]
    print 'BEST PARAMETERS FOR PROBABILITY LESK ALGORITHM ARE:'
    print 'lemmatize=', best_prob_p[1][0], 'remove stopwords=', best_prob_p[1][1], ', window size=', best_prob_p[1][2]
    print '__________________'


    test_base = baseline(test_instances)

    test_lesk = lesk_alg(test_instances, 
                         lemmatize=best_lesk_p[1][0],
                         removeSW=best_lesk_p[1][1],
                         n=best_lesk_p[1][2])

    test_idf = combined(test_instances,
                         useProbs=False,
                         lemmatize=best_idf_p[1][0],
                         removeSW=best_idf_p[1][1],
                         n=best_idf_p[1][2])

    test_prob = combined(test_instances,
                         useProbs=True,
                         lemmatize=best_prob_p[1][0],
                         removeSW=best_prob_p[1][1],
                         n=best_prob_p[1][2])


    print 'FINAL ACCURACIES...'
    print 'Baseline:', getAccuracy(test_base, test_key)
    print 'Lesk:', getAccuracy(test_lesk, test_key)
    print 'IDF Lesk:', getAccuracy(test_idf, test_key) 
    print 'Probability Lesk', getAccuracy(test_prob, test_key)


        
    
