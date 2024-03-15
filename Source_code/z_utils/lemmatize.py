def lemmatize(lemmatizer, text):
    tokens = lemmatizer(text)
    lemmatized_tokens = [token.lemma_ for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text
