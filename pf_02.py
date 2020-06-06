from sklearn.feature_extraction.text import CountVectorizerfrom nltk.corpus import stopwordsimport refrom math import logmain_dir = '/Users/jonathanestrella/Documents/MineriaDeDatos/data/'file = main_dir+'jjer_female1.txt'C = ['H', 'M']def ExtractVocabulary(D):    stop_words = stopwords.words('spanish')    file=D    corpus = []    with open(file, 'r', encoding='utf-8') as content_file:        for line in content_file:            line = line.rstrip().lower()            words = re.findall('[a-záéíóúñ]+', line)            text = ' '.join(word for word in words if word not in stop_words and len(word)>3)            corpus.append(text)              vec = CountVectorizer()        tdm = vec.fit_transform(corpus)        vocabulary = vec.vocabulary_         lista= list(vocabulary.keys())    return(lista)def CountDocsInClass(D, c):    counter = 0    with open(D, 'r', encoding='utf-8') as content_file:        for line in content_file:            if (line[0] == c):                counter += 1    return counterdef ConcaTextOfAllDocsInClass(D, c):    text = ""    with open(D, 'r', encoding = 'utf-8') as i_r:        for line in i_r:            if line[0] == c:                text += line[1:]            text = text.lower()    text = text.replace('\n', ' ')    return textdef CountTokensOfTerm(textc, t):    val = textc.count(t)    return valdef TrainMultinomialNB(C, D):    V = ExtractVocabulary(D)     N = sum(1 for line in open (file, encoding='utf-8')) # Count docs        prior = {'H': 0, 'M': 0}    condprob = []        for c in C:        Nc = CountDocsInClass(D, c)        prior[c] = Nc / N        textc = ConcaTextOfAllDocsInClass(D, c)                for i in range(V):            Tct = CountTokensOfTerm(textc, V[i])            prob_t = (Tct + 1) / sum(Tct )  # Cómo es el denominador? Igonorarlo por mientras            if c == 'H':                condprob[i][0] = prob_t            else:                condprob[i][1] = prob_t                return V, prior, condprob# codigo aún incompleto