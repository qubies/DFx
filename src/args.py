import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--add_json_file", help="add a json file with line by line objects", default=None
)
arg_parser.add_argument(
    "--paraphrase_json_file", help="create a new json file with paraphrases of this one. The input is a json file with line by line objects", default=None
)
arg_parser.add_argument(
    "--generate_max_min_json_file", help="create a max_min from a json file, the input is a json file with line by line objects", default=None
)
arg_parser.add_argument(
    "--json_text_field", help="the json field id as a string", default="full_text",
)
arg_parser.add_argument(
    "--batch_size",
    type=int,
    help="The size of a batch in the file input processor.",
    default=1024,
)

arg_parser.add_argument(
    "--add_phrase", help="a phrase to add to DFx", default=None,
)

arg_parser.add_argument(
    "--num_buckets",
    type=int,
    help="The number of buckets in the search server store. DO NOT CHANGE after initial creation.",
    default=9,
)

arg_parser.add_argument(
    "--search_limit",
    type=int,
    help="The number of results to return with each search",
    default=10,
)


arg_parser.add_argument(
    "--language_model_weight",
    type=float,
    default=5.0,
    help="The amount to count the input of the language model (higher is more)",
)

arg_parser.add_argument(
    "--grammar_weight",
    type=float,
    default=1.0,
    help="The amount to count the input of the grammar checker (higher is more)",
)

arg_parser.add_argument(
    "--length_boost",
    type=float,
    default=0.0,
    help="The amount to increase the score of differing length search results (higher is more)",
)

args = arg_parser.parse_args()

