from config import TERMS, DATA, SEP, CTEXT
from lib import TermData, NLPSource
from corpus.mengzi import fetch_mengzi_full, fetch_mengzi_chapters
from corpus.sep import search_sep, SEP as SEPArticle
from corpus.inpho import is_chinese_philosophy
from cooccurrence.pmi import filter_to_sent_node_lists, build_cooccurrence_network
from nlp.english import tokenize_english_html
from nlp.chinese import tokenize_classical_chinese, STOPWORDS as CHINESE_STOPWORDS
from slugify import slugify
from networkx import Graph
import json

cooccurrence = []

print("Fetching Mengzi full text...")
mengzi = [fetch_mengzi_full()]
print("Fetching Mengzi chapters...")
mengzi = mengzi + fetch_mengzi_chapters()


def get_cooccurence_english(term: str, text: str | list[list[str]]) -> Graph:
    tokens = text if isinstance(text, list) else tokenize_english_html(text)
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=22)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


def get_cooccurence_chinese(term: str, text: str) -> Graph:
    print(f"  [{term}] Tokenizing classical Chinese...")
    tokens = tokenize_classical_chinese(text)

    print(f"  [{term}] {len(tokens)} tokens — building cooccurrence network...")
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=3, stopwords=CHINESE_STOPWORDS)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


def filter_chinese_philosophy(sep_url: str) -> bool:
    if is_chinese_philosophy(sep_url):
        print(" " * 4 + "skipping chinese philosophy search result: " + sep_url)
        return False
    return True


def main() -> None:
    with open(DATA / "terms.json", "w") as file:
        serialized = [{"hanzi": t.hanzi, "english": t.english} for t in TERMS]
        file.write(json.dumps(serialized))

    for term_pairs in TERMS:
        print(f"\n=== {term_pairs.hanzi} / {', '.join(term_pairs.english)} ===")

        # First run NLP for the Chinese term in the Mengzi
        term = term_pairs.hanzi
        data = TermData(
            term,
            sources=[NLPSource(m, cooccurrence=get_cooccurence_chinese(term, m.text))
                     for m in mengzi]
        )
        data.save_json(CTEXT / f"{term}_pmi.json")

        # word_stems: list[str] = [p for stem in term_pairs.renderings for p in stem.patterns]

        # Now iterate for each associated english word stem
        for term in term_pairs.renderings:
            print(f"\n  [{term.label}] Searching SEP...")
            sep_articles = search_sep(
                term, 4, pre_filter=filter_chinese_philosophy)

            print(f"  [{term.label}] {len(sep_articles)} articles found")
            slugified_search_term = "+".join(
                [stem for stem in term.patterns]).replace(' ', '+')
            all_articles = SEPArticle(
                url=f"https://plato.stanford.edu/search/searcher.py?query={slugified_search_term}",
                title="Combined",
                description="Results from all articles",
                text="\n\n".join(a.text for a in sep_articles)
            )
            sep_articles = [all_articles] + sep_articles

            term_count = 0
            stems: set[str] = set()
            sources: list[NLPSource] = []
            for a, article in enumerate(sep_articles):
                tokens = tokenize_english_html(article.text)
                # replace words that match a stem with its canonical label
                canonical: str = term.label
                for i, sent in enumerate(tokens):
                    for j, t in enumerate(sent):
                        if term.matches(t):
                            if a > 1:
                                term_count += 1
                            stems.add(t)
                            tokens[i][j] = canonical
                source = NLPSource(
                    article,
                    cooccurrence=get_cooccurence_english(canonical, tokens)
                )
                sources.append(source)

            print(
                f"  [term.label] found {term_count} occurances accross {len(sep_articles) - 1} articles")
            data = TermData(term.label, stems=list(stems), sources=sources)
            data.save_json(SEP / f"{slugify(term.label)}_pmi.json")


if __name__ == "__main__":
    main()
