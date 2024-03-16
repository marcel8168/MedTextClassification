def lemmatize(lemmatizer, text):
    """
    Lemmatize the input text using the specified lemmatizer.

    Args:
        lemmatizer: SpaCy lemmatizer object.
        text (str): Input text to be lemmatized.

    Returns:
        str: Lemmatized text.
    """
    tokens = lemmatizer(text)
    lemmatized_tokens = [token.lemma_ for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text
