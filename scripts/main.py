from config import TERMS, SEP, CTEXT
from mengzi import fetch_mengzi
from scrape_sep import search_sep
from utils import filter_to_sent_node_lists, build_cooccurrence_network, save_graph_json
from nlp.english import tokenize_english_html
from nlp.chinese import tokenize_classical_chinese, STOPWORDS as CHINESE_STOPWORDS
from slugify import slugify
from networkx import Graph

cooccurrence = []

print("Fetching Mengzi...")
mengzi = fetch_mengzi()
print(f"  Fetched {len(mengzi)} chapters")


def get_cooccurence_english(text: str) -> Graph:
    tokens = tokenize_english_html(text)
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=22)
    return build_cooccurrence_network(
        sent_node_lists, nodes, term, max_nodes=15)


for term_pairs in TERMS:
    print(f"\n=== {term_pairs.hanzi} / {', '.join(term_pairs.english)} ===")

    # First run NLP for the Chinese term in the Mengzi
    term = term_pairs.hanzi
    print(f"  [{term}] Tokenizing classical Chinese...")
    mengzi_full_text = "".join([chapter.text for chapter in mengzi])
    tokens = tokenize_classical_chinese(mengzi_full_text)
    print(f"  [{term}] {len(tokens)} tokens — building cooccurrence network...")
    sent_node_lists, nodes = filter_to_sent_node_lists(
        tokens, term, min_freq=3, stopwords=CHINESE_STOPWORDS)
    G = build_cooccurrence_network(sent_node_lists, nodes, term, max_nodes=15)
    print(
        f"  [{term}] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    save_graph_json(G, CTEXT / f"cooccurrence_{term}.json")
    print(f"  [{term}] Saved → cooccurrence_{term}.json")

    # Now iterate for each associated english term
    for term in term_pairs.english:
        print(f"\n  [{term}] Searching SEP...")
        sep_articles = search_sep(term, 4)
        print(f"  [{term}] {len(sep_articles)} articles found")

        # run NLP on each article
        for article in sep_articles:
            print(f"  [{term}] Processing article: {article.title!r}")
            G = get_cooccurence_english(article.text)
            print(
                f"  [{term}]   → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            title_slug = slugify(article.title)[:12]
            save_graph_json(G, SEP / f"cooccurrence_{term}_{title_slug}.json")

        # run NLP on articles as a whole
        print(f"  [{term}] Building combined graph from all articles...")
        G = get_cooccurence_english("".join([a.text for a in sep_articles]))
        print(
            f"  [{term}] Combined: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        save_graph_json(G, SEP / f"cooccurrence_{term}.json")
        print(f"  [{term}] Saved → cooccurrence_{term}.json")
