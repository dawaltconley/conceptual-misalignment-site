from pathlib import Path
from models import Term, Rendering, Pipeline

# Absolute anchors so the pipelines no longer depend on the current working
# directory. Everything is resolved relative to this file's location.
_SCRIPTS = Path(__file__).resolve().parent   # .../scripts
_ROOT = _SCRIPTS.parent                       # repo root

# --- Inputs (live inside scripts/) ---
MENGZI_DIR = _SCRIPTS / "data" / "mengzi"     # per-chapter Mengzi source .txt
MENGZI_CONLLU = _SCRIPTS / "data" / "mengzi.conllu"

# --- Outputs (live at the repo root) ---
DATA = _ROOT / "src" / "data"
DATA.mkdir(parents=True, exist_ok=True)

PUBLIC = _ROOT / "public"

SEP = PUBLIC / "sep"
SEP.mkdir(parents=True, exist_ok=True)

CTEXT = PUBLIC / "ctext"
CTEXT.mkdir(parents=True, exist_ok=True)

ANALYSIS = _ROOT / "analysis"     # embedding-pipeline artifacts
SEGPOS = _ROOT / "segpos"         # Xunzi segmentation output

EMBEDDINGS = PUBLIC / "embeddings"    # client-loaded scatter datasets
EMBEDDINGS.mkdir(parents=True, exist_ok=True)


# Renderings match a single lowercased token lemma (fnmatch globs). Within/across
# terms, two renderings must not match the same token — the first wins and the
# second becomes a dead node — so patterns are kept disjoint. Multi-word concepts
# match on their head token (e.g. "social norms" -> norm*). Two of the user's
# suggestions were dropped to avoid collisions: 性 "human nature" (collides with
# nature) and 親 "love" (collides with 愛's love).
TERMS: list[Term] = [
    Term('仁', (
        Rendering('benevolence', 'benevolen*'),
        Rendering('humaneness', 'humane*'),
        Rendering('humanity', 'humanit*'),
    )),
    Term('義', (
        Rendering('righteousness', 'righteous*'),
        Rendering('justice', 'justness'),
        Rendering('meaning'),
        Rendering('morality', 'moral*'),
    )),
    Term('禮', (
        Rendering('ritual', 'ritual*'),
        Rendering('propriety'),
        Rendering('etiquette', 'etiquette*'),
        Rendering('mores'),
        # Rendering('social norms', 'social norm*'),
    )),
    Term('智', (
        Rendering('knowledge', 'knowledge*'),
        Rendering('wisdom', 'wisdom*', 'wise*'),
        Rendering('intelligence', 'intelligen*'),
    )),
    Term('信', (
        Rendering('trustworthiness', 'trustworth*'),
        Rendering('faith', 'faith*'),
        Rendering('sincerity', 'sincer*'),
    )),
    # Term('性', (
    #     Rendering('nature', 'nature*', 'human nature'),
    #     Rendering('innateness', 'innate*'),
    #     Rendering('character', 'character*'),
    #     Rendering('predisposition', 'predispos*'),
    # )),
    # Term('心', (
    #     Rendering('heart', 'heart', 'hearts', 'heartfelt'),
    #     Rendering('mind', 'mind', 'minds', 'mindful*'),
    #     Rendering('heartmind', 'heartmind*', 'heart-mind*'),
    #     Rendering('feeling', 'feeling*'),
    # )),
    # Term('親', (
    #     Rendering('parents', 'parent*'),
    #     Rendering('kin', 'kin', 'kinship*'),
    #     Rendering('intimates', 'intimate*', 'intimacy'),
    #     Rendering('affection', 'affection*'),
    # )),
    # Term('愛', (
    #     Rendering('care', 'care*'),
    #     Rendering('cherish', 'cherish*'),
    #     Rendering('love', 'love*'),
    #     Rendering('pity', 'pity', 'piti*'),
    # )),
    # Term('道', (
    #     Rendering('way', 'ways'),
    #     Rendering('doctrine', 'doctrine*'),
    #     Rendering('method', 'method*'),
    #     Rendering('path', 'path*'),
    #     Rendering('speak', 'speaks', 'speaking', 'spoke*'),
    # )),
    # Term('德', (
    #     Rendering('virtue', 'virtue*'),
    #     Rendering('power', 'power*'),
    #     Rendering('charisma', 'charisma*'),
    # )),
]

CHINESE_STOPWORDS: set[str] = {
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已",
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    "我", "吾", "汝", "其", "此", "彼", "是",
    "有", "無", "為", "爲", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    "然", "得", "能", "可", "將", "及", "皆", "未", "與",
}

with open(_SCRIPTS / "stopwords" / "english.conf", "r") as file:
    ENGLISH_STOPWORDS = set[str]()
    for line in file:
        line = line.strip()
        if line and not line.startswith('#'):
            ENGLISH_STOPWORDS.add(line)

# Surface form -> corrected lemma, patching en_core_web_sm's mistakes before any
# downstream code sees them (applied in corpus.parse). Keyed on the surface, not
# the wrong lemma, because the wrong lemma is often a real word too — see the
# conf file's header.
with open(_SCRIPTS / "lemmas" / "english.conf", "r") as file:
    ENGLISH_LEMMAS = dict[str, str]()
    for line in file:
        line = line.split('#')[0].strip()
        if not line:
            continue
        surface, _, lemma = line.partition('->')
        ENGLISH_LEMMAS[surface.strip()] = lemma.strip()

MENGZI_PIPELINE = Pipeline(
    model="hsc748NLP/GujiRoBERTa_fan",
    out_dir=CTEXT,
    min_freq=5,
    cooccurrence_min_freq=3,
    content_pos=frozenset({"NOUN", "VERB", "ADJ"}),
    stopwords=frozenset(CHINESE_STOPWORDS),
    debias="abtt",
)

SEP_PIPELINE = Pipeline(
    model="roberta-base",
    out_dir=SEP,
    min_freq=20,
    cooccurrence_min_freq=10,
    content_pos=frozenset({"NOUN", "VERB", "ADJ"}),
    stopwords=frozenset(ENGLISH_STOPWORDS),
    merge_variants=True,
    debias="abtt",
)

# # for reference, not currently used for filtering
# NER_EXCLUDES = {
#     'PERSON',  # gets some people not picked up by propn
#     'DATE',
#     # "geopolitical entity," includes some famous philosophers (e.g. Descartes)
#     'GPE',
#     'ORDINAL',  # usually a century
#     'ORG',  # often misidentified people
#     'NORP',  # philosophical schools
# }
