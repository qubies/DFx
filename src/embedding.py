import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import json
from tqdm import tqdm
from collections import namedtuple, OrderedDict
import math
import re
from myutils import (
    print_banner_completion_wrapper,
    print_banner,
    remove_links,
    cosine_similarity,
)

BATCH_SIZE = 1024

# globals
u = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


def load_ditstilbert():
    global distilbert_tokenizer
    global distilbert_model
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    if distilbert_model == None:
        distilbert_model = DistilBertForMaskedLM.from_pretrained(
            "distilbert-base-uncased", output_hidden_states=True
        )
        distilbert_model.eval()
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )


def load_gpt():
    global gpt_tok, gpt_model
    if not gpt_model:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        device = "cpu"
        model_id = "gpt2-large"
        gpt_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        gpt_tok = GPT2TokenizerFast.from_pretrained(model_id)


def load_tool():
    global tool
    if not tool:
        tool = language_tool_python.LanguageTool("en-US")


def distilbert_embed(sentence):
    load_ditstilbert()
    inputs = distilbert_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        # hidden:layer:token:768
        emb = distilbert_model(**inputs)[1][-2][0]
        return torch.mean(emb, dim=0).numpy()


# modified from https://github.com/huggingface/transformers/issues/37
def language_score(sentence, lm_weight):
    load_gpt()
    load_tool()
    if len(sentence.strip().split()) <= 1:
        return 1 / 10000  # error
    encoding = gpt_tok(sentence, return_tensors="pt")
    if len(encoding) > 512:
        return 1 / 10000  # error

    input_ids = encoding.input_ids.to(device)
    with torch.no_grad():
        outputs = gpt_model(input_ids, labels=input_ids)
        log_likelihood = math.exp(float(outputs[0].sum() / len(input_ids)))
    return (1 / log_likelihood) * lm_weight


def grammar_score(sentence, grammar_weight):
    grammar_mistakes = len(tool.check(sentence))
    return -1 * (grammar_mistakes * grammar_weight)


def fix_grammar(sentence):
    load_tool()
    return tool.correct(sentence)


def use_embed(s):
    return u([s]).numpy()[0]


def use_batch_embed(l):
    return u(l).numpy()


def bucket_compare(test_phrase, filename, k, debug=False):
    a = embed(test_phrase)
    results = []
    batch = []
    Result = namedtuple("Result", ["Cosine", "Dot", "Phrase"])
    phrase_set = set()
    with open(filename) as f:
        for line in tqdm(f, total=1000000):
            raw = json.loads(line)
            if raw["full_text"] not in phrase_set:
                phrase_set.add(raw["full_text"])
                batch.append(remove_links(raw["full_text"], token="URL"))
            if len(batch) >= BATCH_SIZE:
                vecs = u.batch_embed(batch)
                for i, vec in enumerate(vecs):
                    results.append(
                        Result(
                            cosine_similarity(vec, a),
                            np.dot(a, vec),
                            batch[i],
                        )
                    )
                batch = []

    if len(batch) >= BATCH_SIZE:
        vecs = u.batch_embed(batch)
        for i, vec in enumerate(vecs):
            results.append(Result(cosine_similarity(vec, a), np.dot(a, vec), batch[i]))

    results.sort(key=lambda x: x.Cosine, reverse=True)
    cosine_set = set()

    if debug:
        print_banner("Cosine Similarity")
    for i in range(k):
        if debug:
            print(f"Score: {results[i].Cosine}, Phrase: {results[i].Phrase}")
        cosine_set.add(results[i].Phrase.strip())

    for bucket in [2] + [x for x in range(5, 61, 5)]:
        dfx_set = set()
        with open(f"results/{test_phrase}_{bucket}") as f:
            count = 0
            for line in f:
                if count > k:
                    break
                count += 1
                o = json.loads(line)
                for phrase in o["phrases"]:
                    dfx_set.add(phrase.strip())

        only_in_cosine = cosine_set - dfx_set
        only_in_dfx = dfx_set - cosine_set
        diff = (only_in_cosine) | (only_in_dfx)
        if debug:
            print_banner("Reslts")
        if debug:
            print(f"Total Differences: {len(diff)}")
        if debug:
            print_banner(f"Only in cosine: {len(only_in_cosine)}")
        for line in only_in_cosine:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        if debug:
            print_banner(f"Only in dfx: {len(only_in_dfx)}")
        for line in only_in_dfx:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        shared = cosine_set & dfx_set
        if debug:
            print_banner(f"Shared: {len(shared)}")
        for line in shared:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        if not debug:
            print(f"{bucket}, {len(diff)}, {len(only_in_cosine)}, {len(only_in_dfx)}")


def top_k(test_phrase, filename, k, debug=False):
    a = embed(test_phrase)
    results = []
    batch = []
    Result = namedtuple("Result", ["Cosine", "Dot", "Phrase"])
    phrase_set = set()
    with open(filename) as f:
        for line in tqdm(f, total=1000000):
            raw = json.loads(line)
            if raw["full_text"] not in phrase_set:
                phrase_set.add(raw["full_text"])
                batch.append(remove_links(raw["full_text"], token="URL"))
            if len(batch) >= BATCH_SIZE:
                vecs = u.batch_embed(batch)
                for i, vec in enumerate(vecs):
                    results.append(
                        Result(
                            cosine_similarity(vec, a),
                            np.dot(a, vec),
                            batch[i],
                        )
                    )
                batch = []

    if len(batch) >= BATCH_SIZE:
        vecs = u.batch_embed(batch)
        for i, vec in enumerate(vecs):
            results.append(Result(cosine_similarity(vec, a), np.dot(a, vec), batch[i]))

    results.sort(key=lambda x: x.Cosine, reverse=True)
    for x in range(1, k + 1):
        cosine_set = set()
        dfx_set = set()

        if debug:
            print_banner("Cosine Similarity")
        for i in range(x):
            if debug:
                print(f"Score: {results[i].Cosine}, Phrase: {results[i].Phrase}")
            cosine_set.add(results[i].Phrase.strip())

        with open(f"results/{test_phrase}") as f:
            count = 1
            for line in f:
                o = json.loads(line)
                for phrase in o["phrases"]:
                    dfx_set.add(phrase.strip())
                if count == x:
                    break
                count += 1

        only_in_cosine = cosine_set - dfx_set
        only_in_dfx = dfx_set - cosine_set
        diff = (only_in_cosine) | (only_in_dfx)
        if debug:
            print_banner("Reslts")
        if debug:
            print(f"Total Differences: {len(diff)}")
        if debug:
            print_banner(f"Only in cosine: {len(only_in_cosine)}")
        for line in only_in_cosine:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        if debug:
            print_banner(f"Only in dfx: {len(only_in_dfx)}")
        for line in only_in_dfx:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        shared = cosine_set & dfx_set
        if debug:
            print_banner(f"Shared: {len(shared)}")
        for line in shared:
            if debug:
                print(line)
            if debug:
                print("+++++-----")
        if not debug:
            print(f"{x}, {len(diff)}, {len(only_in_cosine)}, {len(only_in_dfx)}")
