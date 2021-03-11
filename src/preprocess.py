from spacy.matcher import Matcher
import json
from string import punctuation
import re
from myutils import remove_links

repeat_pattern = re.compile(r"([0-9a-zA-Z])\1{2,}")
nlp = None
letters = re.compile("[a-zA-Z]")
vp_pattern = [
    {"POS": "VERB", "OP": "?"},
    {"POS": "ADV", "OP": "*"},
    {"POS": "AUX", "OP": "*"},
    {"POS": "VERB", "OP": "+"},
]
matcher = None

STOPWORDS = set()
#  with open("src/resources/stopwords.txt") as f:
with open("src/resources/typical_stopwords.txt") as f:
    for line in f:
        STOPWORDS.add(line.lower().strip())


def simplify(phrase):
    result = []
    prev = ""
    phrase = repeat_pattern.sub(r"\1\1", phrase).strip(punctuation).strip()
    for word in phrase.lower().split(" "):
        if word != prev:
            prev = word
            if word not in STOPWORDS:
                result.append(word)
    return " ".join(result)


def check_chunk(chunk):
    chunk = repeat_pattern.sub(r"\1\1", remove_links(chunk)).strip(punctuation).strip().replace("'","")
    if not letters.search(chunk):
        return None
    if len(chunk) < 1:# or len(chunk) > 35:
        return None
    return chunk


def add_if_ok(phrase, phrases):
    s = check_chunk(phrase[0])
    if s:
        phrases.add((s, phrase[1], phrase[2]))


def spacy_from_offsets(doc, start, end, exclusions=[]):
    return (
        "".join(text.text_with_ws for i, text in enumerate(doc[start:end]) if i not in exclusions),
        doc[start].idx,
        doc[end - 1].idx + len(doc[end - 1].text),
    )


def lookahead(thing):
    try:
        iterable = iter(thing)
        current = next(iterable)
    except StopIteration:
        return
    for val in iterable:
        yield current, False
        current = val
    yield current, True


def verb_splitter(s):
    global nlp, matcher
    if nlp == None:
        import spacy
        from spacy.tokens import Doc

        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)
        matcher.add("Verb phrase", None, vp_pattern)

    for doc in nlp.pipe(s, disable=["ner"]):
        for sentence in doc.sents:
            phrases = set()
            for vp, last in lookahead(matcher(sentence.as_doc())):
                _, start, end = vp
                add_if_ok(spacy_from_offsets(doc, 0, start), phrases)
                add_if_ok(spacy_from_offsets(doc, 0, end), phrases)
                add_if_ok(spacy_from_offsets(doc, start, end), phrases)
                if last and len(doc) != end:
                    add_if_ok(spacy_from_offsets(doc, 0, len(doc)), phrases)

            if len(phrases) > 0:
                yield sorted(phrases, key=lambda x: len(x[0]))
            else:
                text = check_chunk("".join(text.text_with_ws for text in sentence))
                if text:
                    yield [(text, 0, len(text))]
                else:
                    yield []


def phrase_chunk(s):
    global nlp, matcher
    if nlp == None:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)
        matcher.add("Verb phrase", None, vp_pattern)

    #  for i, z in enumerate(s[0]):
    #      print(f"{i}:{z}")

    for doc in nlp.pipe(s, disable=["ner"]):
        phrases = set()

        ## collect the noun phrases
        for np in doc.noun_chunks:
            new_phrase = check_chunk(np.text)
            if new_phrase:
                phrases.add(
                    (
                        new_phrase,
                        doc[np.start].idx,
                        doc[np.end - 1].idx + len(doc[np.end - 1].text),
                    )
                )

        ## collect the subtree chunks
        for token in doc:
            s = check_chunk(
                #  f"""{"".join([tok.text_with_ws for tok in doc[token.left_edge.i : token.right_edge.i + 1]])}"""
                f"""{"".join([tok.text_with_ws for tok in token.subtree])}"""
            )
            if s:

                phrases.add(
                    (
                        s,
                        token.left_edge.idx,
                        token.right_edge.idx + len(token.right_edge.text),
                    )
                )

        ## collect the verb phrases
        for vp in matcher(doc):
            _, start, end = vp
            s = check_chunk("".join(text.text_with_ws for text in doc[start:end]))
            if s:
                phrases.add(
                    (s, doc[start].idx, doc[end - 1].idx + len(doc[end - 1].text))
                )
        yield phrases

tokenizer = None
#   all_words = set()
#   with open("vocabulary", encoding="ISO-8859-1") as f:
#       for line in f:
#           all_words.add(line.strip())
punct_re = re.compile(r'[^\w\s]')

def running_chunk(s):
    global nlp, tokenizer
    if tokenizer == None:
        import spacy
        from spacy.tokenizer import Tokenizer
        nlp = spacy.load("en_core_web_sm")
        tokenizer = Tokenizer(nlp.vocab)

    for sentence in s:

        phrases = set()
        doc = tokenizer(sentence)
        exclusions = []
        for i, tok in enumerate(doc):
            add_if_ok(spacy_from_offsets(doc, 0, i+1, exclusions), phrases)
    
        yield phrases


def simple_chunk(s):
    for sentence in s:
        words = sentence.split()
        phrases = set()
        for w1, w2, w3 in zip(words, words[1:], words[2:]):
            phrases.add(f"{w1} {w2} {w3}".strip())
        yield (phrases)


if __name__ == "__main__":
    while 1:
        print(list(running_chunk([input("Test input:")])))
