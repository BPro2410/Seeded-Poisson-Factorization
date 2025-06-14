import re

def clean(text, remove_punct = False):
    """
    Data (text) cleaning.
    See: https://www.w3schools.com/python/python_regex.asp for regex pattern explanations.

    Args:
        text (str): Text to be cleaned

    Return:
        str: Cleaned text
    """
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)

    if remove_punct == True:
        # Remove punctuation: '!hi. wh?at is the weat[h]er lik?e.' -> 'hi what is the weather like'
        text = re.sub(r'[^\w\s]', '', text)

    return text.lower().strip()

