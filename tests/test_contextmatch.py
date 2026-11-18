from raglab.contextmatch import DocTokenIndex, contexts_include_gold


def test_token_match_ignores_a_missing_space_and_finds_the_gold_doc():
    documents = [
        ("gold", "A predicted network. Collectively, this work expands peptide signalling."),
        ("other", "Unrelated abstract about glaciers and firn cores."),
    ]
    index = DocTokenIndex(documents)
    # The recursive splitter dropped the space after the period.
    context = "network.Collectively, this work expands peptide signalling."
    assert index.match(context) == ["gold"]
    assert contexts_include_gold([context], {"gold"}, index)
    assert not contexts_include_gold(["firn cores"], {"gold"}, index)
    assert contexts_include_gold(["firn cores"], {"other"}, index)
