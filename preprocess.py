def remove_noise(tweets):
    """
    generate train and test data for nltk sentiment classifier
    
    Parameters
    ----------
    tweets|string
        text of the tweet(s) in string format
    Returns
    -------
    cleaned tweets|string
        cleaned text of tweets
    Example
    -------
    >>>from abid import remove_noise
    >>>cleaned_text= remove_noise(text)
    """
    import nltk
    import random,re,string
    import pandas as pd
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    stopwords = nltk.corpus.stopwords.words('english')
    try:
        tweet_tokens = nltk.tokenize.word_tokenize(tweets)
        cleaned_tokens = []
        for token, tag in nltk.tag.pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stopwords:
                cleaned_tokens.append(token.lower())
        cleaned_tokens = [i.strip() for i in cleaned_tokens if '.com' not in i and 'bit.ly' not in i]
        text = ''
        for i in cleaned_tokens:
            if i.isalpha():
                text+=i+' '
        return text
    except:
        return tweets

def get_train_test_twitter(test_size = .3):
    """
    generate train and test data for nltk sentiment classifier
    
    Parameters
    ----------
    test_size|float
        test_size is float number ranging from 0.0 to 1.0. determines the size of train and test data.
        default is 0.3
    Returns
    -------
    train_data,test_data
        list of dictionaries consisting tokenized and labeled tweets for train and test data
    Example
    -------
    >>>from abid import get_train_test_twitter
    >>>train,test = get_train_test_twitter(test_size = .2)
    """
    import nltk
    import random,re,string
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('twitter_samples')
    positive_tweet_tokens = nltk.corpus.twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = nltk.corpus.twitter_samples.tokenized('negative_tweets.json')
    
    def remove_noise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in nltk.tag.pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens
    
    stop_words = nltk.corpus.stopwords.words('english')
    positive_cleaned_tokens_list = [remove_noise(token,stop_words) for token in positive_tweet_tokens]
    negative_cleaned_tokens_list = [remove_noise(token,stop_words) for token in negative_tweet_tokens]
    
    def tweets_for_model(cleaned_tokens_list):
        tokens = []
        for tweet_token in cleaned_tokens_list:
            tweet_dict = {}
            for token in tweet_token:
                tweet_dict[token] = True
            tokens.append(tweet_dict)
        return tokens
    
    positive_tokens_model = tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_model = tweets_for_model(negative_cleaned_tokens_list)
    positive_dataset = [(tweet_dict,1) for tweet_dict in positive_tokens_model]
    negative_dataset = [(tweet_dict,0) for tweet_dict in negative_tokens_model]
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    t_size = test_size*len(dataset)
    tr_size = len(dataset)-t_size
    train_data = dataset[:int(tr_size)]
    test_data = dataset[int(t_size):]
    
    return train_data, test_data

def preprocess_text(texts):
    """
    preprocess text data
    
    Parameters
    ----------
    texts|list
        list of texts
    Returns
    -------
    list of dict
        list of dictionaries consisting tokenized texts
    Example
    -------
    >>>from abid import preprocess_text
    >>>text = "i will tell you that you guys need to hold on GOOGl for now"
    >>>preprocessed_text = preprocess_text(text)
    """
    import nltk
    import random,re,string
    import pandas as pd
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('twitter_samples')
    nltk.download('punkt')
    
    def remove_noise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in nltk.tag.pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens


    def tweets_for_model(cleaned_tokens_list):
        tokens = []
        for tweet_token in cleaned_tokens_list:
            tweet_dict = {}
            for token in tweet_token:
                tweet_dict[token] = True
            tokens.append(tweet_dict)
        return tokens

    stop_words = nltk.corpus.stopwords.words('english')
    datas = []
    if type(texts) == list:
        for i in texts:
            if not pd.isnull(i):
                tokens = [nltk.tokenize.word_tokenize(i)]
                cleaned_tokens_list = [remove_noise(token,stop_words) for token in tokens]
                tokens_model = tweets_for_model(cleaned_tokens_list)
                dataset = [(tweet_dict) for tweet_dict in tokens_model]
                datas.append(dataset[0])
            else:
                datas.append(i)
    elif type(texts)== str:
        tokens = [nltk.tokenize.word_tokenize(texts)]
        cleaned_tokens_list = [remove_noise(token,stop_words) for token in tokens]
        tokens_model = tweets_for_model(cleaned_tokens_list)
        datas = [(tweet_dict) for tweet_dict in tokens_model]
    return datas

def predict(model,data):
    """
    predict tokenized text data
    
    Parameters
    ----------
    model
        machine learning model used to classify text
    data|list
        list of dictionaries of tokenized texts or simply a list after doing abid.preprocess_text(text)
    Returns
    -------
    list
        list predictions
    Example
    -------
    >>>from abid import preprocess_text
    >>>from abid import predict
    >>>text = "i will tell you that you guys need to hold on GOOGl for now"
    >>>preprocessed_text = preprocess_text(text)
    >>>pred = predict(preprocessed_text)
    """
    import pandas as pd
    predicted = []
    if len(data) != 1:
        for tokens in data:
            if not pd.isnull(tokens):
                predicted.append(model.classify(tokens))
            else:
                predicted.append(tokens)
    elif len(data) == 1:
        if not pd.isnull(data[0]):
            predicted = model.classify(data[0])
    return predicted
