from . import *

# Totals and limits
TOTAL_PMLA = 71902
TOTAL_JSTOR = 12412004
TOTAL_JSTOR_DATA = 32783
SLICE_LEN = 1000
FEAT_N = 100
FEAT_MIN_COUNT = 0
MIN_YEAR = 1925
MAX_YEAR = 2025
CONTEXT_LEN = 15
CLASSIFY_BY_FEAT_SAMPLE_SIZE = 10_000

# Bad features to exclude
BAD_SLICE_FEATS = [
    "sent_num_punct",
    "sent_num_punct_colon",
    "sent_num_punct_comma",
    "sent_num_parens",
    "sent_num_words_verb",
    "sent_num_words_noun",
    "sent_num_words_adverb",
    "sent_num_words_adjective",
    "deprel_list",
    "pos_LS",
    "pos_ADD",
    "num_words",
    "num_recog_words",
    "deprel_obl:tmod",
    "pos_''",
    # "pos_,",
    "pos_``",
    # "pos_.",
    "pos_$",
    "ttr",
    "ttr_recog",
]
BAD_POS = {"X"}
BAD_DEPREL = {"flat"}

# Paths
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_HERE, "..", "data")
PATH_METADATA = os.path.join(PATH_DATA, "metadata.csv")
PATH_TXT = os.path.join(PATH_DATA, "txt")
FN_PMLA = os.path.join(PATH_DATA, "raw/LitStudiesJSTOR.jsonl")
FN_JSTOR = os.path.join(PATH_DATA, "raw/jstor_metadata_2025-11-28.jsonl.gz")
FN_JSTOR_DATA = os.path.join(PATH_DATA, "raw/jstor_data.jsonl.gz")
PATH_NON_CONTENT_WORDS = os.path.join(PATH_DATA, "non_content_words.xlsx")
PATH_MDW_DATA = os.path.join(PATH_DATA, "mdw_data.csv")
PATH_WORDFREQS_TSV = os.path.join(PATH_DATA, "raw", "word_freqs.tsv")
PATH_WORDFREQS_TSV_NL = 62141323
PATH_WORDDB = os.path.join(PATH_DATA, "raw", "worddb.byu.txt")
PATH_TOTAL_TEXT_COUNTS = os.path.join(PATH_DATA, "total_text_counts.json")
PATH_FEAT_WEIGHTS = os.path.join(PATH_DATA, "feats_perspectival2.xlsx")

# Stashes
NLP_STASH = HashStash("osp_nlp")
PMLA_STASH = HashStash("osp_pmla")
JSTOR_STASH = HashStash("osp_jstor")
DF_STASH = HashStash("osp_df", serializer="pickle")
STASH_WORD_FREQS = HashStash("osp_word_freqs", append_mode=False)
STASH_POS_COUNTS = HashStash("osp_slices_1000_pos_counts")
STASH_FEAT2WORD2COUNT = HashStash("osp_slices_1000_feat2word2count")
STASH_FEAT2WORD2EG = HashStash("osp_slices_1000_feat2word2eg")
STASH_SENT_FEAT_COUNTS = HashStash("osp_slices_1000_sent_feat_counts")
STASH_SLICES = HashStash("osp_slices_1000")
STASH_SLICES_NLP = HashStash("osp_slices_1000_nlp")
STASH_SLICE_FEATS = HashStash("osp_slices_1000_feats")
STASH_COUNTS = HashStash("osp_counts")
STASH_PREDS_FEATS = HashStash('osp_preds_feats')

NORMALIZE_DATA = False

# Wordsets
WORDSETS = ["top", "content", "non_content"]

# Global variables
NLP = None
METADATA = None
OK_WORDS = None

# Non-content POS tags
NON_CONTENT_POS = [
    "pp",  # Personal pronoun
    "pn",  # Indefinite pronoun
    "appge",  # Possessive pronoun/determiner
    "at",  # Article
    "dd",  # General determiner
    "da",  # After-determiner
    "db",  # Before-determiner
    "ii",  # General preposition
    "cs",  # Subordinating conjunction
    "cc",  # Coordinating conjunction
    "to",  # Infinitive marker
    "xx",  # Negative particle
    "ex",  # Existential 'there'
    "ge",  # Genitive marker
    "uh",  # Interjection
]

# Auxiliary verbs
AUXILIARY_VERBS = [
    "vm",  # Modal auxiliary
    "vb0",  # Base form of 'be'
    "vbz",  # 'is'
    "vbm",  # 'am'
    "vbr",  # 'are'
    "vbdz",  # 'was'
    "vbdr",  # 'were'
    "vbg",  # 'being'
    "vbn",  # 'been'
    "vh0",  # Base form of 'have'
    "vhz",  # 'has'
    "vhd",  # 'had'
    "vd0",  # Base form of 'do'
    "vdz",  # 'does'
    "vdd",  # 'did'
]

# POS descriptions
POS2DESC = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
    "-RRB-": "Right parenthesis",
    "-LRB-": "Left parenthesis",
    "-RSB-": "Right square bracket",
    "-LSB-": "Left square bracket",
    "NFP": "Superfluous punctuation",
    "HYPH": "Hyphen",
    "GW": "'Goes With' or error marker",
    "AFX": "Detached prefix or suffix",
}

# Dependency relation descriptions
DEP2DESC = {
    "acl": "Clausal modifier of noun (adnominal clause)",
    "acl:relcl": "Relative clause modifier",
    "advcl": "Adverbial clause modifier",
    "advcl:relcl": "Adverbial relative clause modifier",
    "advmod": "Adverbial modifier",
    "advmod:emph": "Emphasizing word, intensifier",
    "advmod:lmod": "Locative adverbial modifier",
    "amod": "Adjectival modifier",
    "appos": "Appositional modifier",
    "aux": "Auxiliary",
    "aux:pass": "Passive auxiliary",
    "case": "Case marking",
    "cc": "Coordinating conjunction",
    "cc:preconj": "Preconjunct",
    "ccomp": "Clausal complement",
    "clf": "Classifier",
    "compound": "Compound",
    "compound:lvc": "Light verb construction",
    "compound:prt": "Phrasal verb particle",
    "compound:redup": "Reduplicated compounds",
    "compound:svc": "Serial verb compounds",
    "conj": "Conjunct",
    "cop": "Copula",
    "csubj": "Clausal subject",
    "csubj:outer": "Outer clause clausal subject",
    "csubj:pass": "Clausal passive subject",
    "dep": "Unspecified dependency",
    "det": "Determiner",
    "det:numgov": "Pronominal quantifier governing the case of the noun",
    "det:nummod": "Pronominal quantifier agreeing in case with the noun",
    "det:poss": "Possessive determiner",
    "discourse": "Discourse element",
    "dislocated": "Dislocated elements",
    "expl": "Expletive",
    "expl:impers": "Impersonal expletive",
    "expl:pass": "Reflexive pronoun used in reflexive passive",
    "expl:pv": "Reflexive clitic with an inherently reflexive verb",
    "fixed": "Fixed multiword expression",
    "flat": "Flat expression",
    "flat:foreign": "Foreign words",
    "flat:name": "Names",
    "goeswith": "Goes with",
    "iobj": "Indirect object",
    "list": "List",
    "mark": "Marker",
    "nmod": "Nominal modifier",
    "nmod:poss": "Possessive nominal modifier",
    "nmod:tmod": "Temporal modifier",
    "nsubj": "Nominal subject",
    "nsubj:outer": "Outer clause nominal subject",
    "nsubj:pass": "Passive nominal subject",
    "nummod": "Numeric modifier",
    "nummod:gov": "Numeric modifier governing the case of the noun",
    "obj": "Object",
    "obl": "Oblique nominal",
    "obl:agent": "Oblique agent in passive construction",
    "obl:arg": "Oblique argument",
    "obl:lmod": "Locative modifier",
    "obl:tmod": "Temporal modifier",
    "orphan": "Orphan",
    "parataxis": "Parataxis",
    "punct": "Punctuation",
    "reparandum": "Overridden disfluency",
    "root": "Root",
    "vocative": "Vocative",
    "xcomp": "Open clausal complement",
    "det:predet": "Predeterminer",
    "nmod:unmarked": "Unmarked nominal modifier",
    "obl:unmarked": "Unmarked Oblique Nominal",
}

SENTFEAT2DESC={
    'num_words': 'Number of words in sentence',
    'num_words_in_independent_clauses': 'Number of words in independent clauses',
    'num_words_in_dependent_clauses': 'Number of words in dependent clauses',
    'height': 'Height of sentence syntax trees',
    'num_independent_clauses': 'Number of independent clauses',
    'num_dependent_clauses': 'Number of dependent clauses',
}

# Combined feature descriptions
FEAT2DESC = {
    **{f'pos_{x}':v for x,v in POS2DESC.items()},
    **{f'deprel_{x}':v for x,v in DEP2DESC.items()},
    **{f'sent_{x}':v for x,v in SENTFEAT2DESC.items()},
}

stashed_result = HashStash("osp_stashed_result").stashed_result

# POS names mapping
pos_names = {"N": "noun", "V": "verb", "J": "adjective", "R": "adverb"}


COMPARISONS = [
    (
        ('1925-1950 Philosophy', 'discipline=="Philosophy" & 1925<=year<1950'),
        ('1925-1950 Literature', 'discipline=="Literature" & 1925<=year<1950'),
    ),
    (
        ('1950-1975 Philosophy', 'discipline=="Philosophy" & 1950<=year<1975'),
        ('1950-1975 Literature', 'discipline=="Literature" & 1950<=year<1975'),
    ),
    (
        ('1975-2000 Philosophy', 'discipline=="Philosophy" & 1975<=year<2000'),
        ('1975-2000 Literature', 'discipline=="Literature" & 1975<=year<2000'),
    ),
    (
        ('2000-2025 Philosophy', 'discipline=="Philosophy" & 2000<=year<2025'),
        ('2000-2025 Literature', 'discipline=="Literature" & 2000<=year<2025'),
    ),
]
GROUPS_TRAIN = COMPARISONS[0]