import json

import click
from transformers import CLIPTokenizer

from pydiffuser.clip import tokenize as clip_tokenize


@click.group()
def cli():
    pass


@cli.command()
@click.argument("text")
@click.option(
    "--tokens",
    type=click.Path(dir_okay=False, writable=True),
    default="tokens.json",
    help="Path to save the tokens JSON to.",
)
@click.option(
    "--mappings",
    type=click.Path(dir_okay=False, writable=True),
    default="mappings.json",
    help="Path to save the mappings JSON to.",
)
@click.option(
    "--tokenizer",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to a custom CLIP tokenizer.",
)
def tokenize(text, tokens, mappings, tokenizer):
    """Tokenizes the given text with a CLIP tokenizer."""

    clip_tokenizer = None
    if tokenizer:
        clip_tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
    token_lists, mapping_lists = clip_tokenize(text, clip_tokenizer=clip_tokenizer)
    with open(tokens, "w") as f:
        json.dump(token_lists, f)
    with open(mappings, "w") as f:
        json.dump(mapping_lists, f)


if __name__ == "__main__":
    cli()
