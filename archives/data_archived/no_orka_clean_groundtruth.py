import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import inflect
from nltk.corpus import wordnet as wn
import nltk
from difflib import get_close_matches
nltk.download('wordnet')
nltk.download('omw-1.4')
p = inflect.engine()

def bert_based_clustering():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    canonical_colors = {
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "orange": "orange",
        "purple": "purple",
        "pink": "pink",
        "brown": "brown",
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",  # UK/US spelling
        "silver": "silver",
        "gold": "gold",
        "beige": "beige",
        "cream": "cream",
        "tan": "tan",
        "navy": "navy",
        "maroon": "maroon",
        "burgundy": "burgundy",
        "bronze": "bronze",
        "charcoal": "gray",
        "clear": "transparent",
        "transparent": "transparent",
        "colorless": "transparent",
        "multicolor": "multicolored",
        "multi-colored": "multicolored",
        "multi": "multicolored",
        "rainbow": "multicolored",
        "patterned": "patterned",
        "natural wood": "brown",
        "wood": "brown",
        "metallic": "silver",
        "stainless steel": "silver",
        "ceramic": "white",
        "rose gold": "gold",
        "off-white": "white",
        "dark blue": "blue",
        "light blue": "blue",
        "light gray": "gray",
        "dark gray": "gray",
        "dark green": "green",
        "light green": "green",
        "reddish-brown": "brown",
        "golden": "gold",
        "blond": "yellow",
        "ginger": "orange",
        "chrome": "silver",
        "camo": "multicolored",
        "varied": "multicolored",
        "various": "multicolored",
        "custom": "multicolored",
    }
    color_labels = list(canonical_colors.keys())
    color_embeddings = model.encode(color_labels, convert_to_tensor=True)

    def extract_colors_bert(text):
        if pd.isna(text):
            return []
        emb = model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(emb, color_embeddings)[0]
        matched = [canonical_colors[color_labels[i]] for i, score in enumerate(scores) if score > 0.5]
        return sorted(set(matched))

def tokenize_clean(text):
    """Tokenize input text and filter to alphanumeric, lowercase tokens"""
    if pd.isna(text):
        return []
    # Split on commas, semicolons, slashes, "and", etc.
    tokens = re.split(r'[,\;/]| and | or |\n', str(text))
    return [t.strip().lower() for t in tokens if t.strip().isalpha()]

def clean_texture(text):
    if pd.isna(text):
        return []
    canonical_map = {
        "glossy": "glossy", "shiny": "glossy",
        "rough": "rough", "coarse": "rough", "bumpy": "rough", "gritty": "rough",
        "smooth": "smooth", "glassy": "smooth",
        "soft": "soft", "fuzzy": "soft", "fluffy": "soft", "plush": "soft",
        "matte": "matte", "textured": "textured", "ribbed": "textured", "grippy": "textured",
        "slippery": "slippery", "sticky": "sticky", "hard": "hard", "firm": "hard",
        "wet": "wet", "dry": "dry",
        "silky": "silky", "woolly": "woolly", "leathery": "leathery"
    }
    # Step 1: normalize
    text = text.lower()
    text = re.sub(r"[\.;\-]", ",", text)
    text = re.sub(r"\s+", " ", text)

    # Step 2: extract words before/inside parentheses or separated by commas
    tokens = re.findall(r'\b\w+\b', text)  # basic tokenization
    candidates = set()

    for token in tokens:
        if token in canonical_map:
            candidates.add(canonical_map[token])
    return sorted(candidates)

def clean_and_split(text):
    # Remove unwanted fragments and parentheticals
    text = re.sub(r'\b(e\.g\.|etc\.|or)\b', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.replace(")", "").replace(".", "")

    # Force space before in/on to allow splitting
    text = re.sub(r"\b(on|in)\b", r" \1", text)

    # Split by commas or known prepositions
    raw_items = re.split(r",| and | on | in ", text)

    cleaned = []
    for item in raw_items:
        item = item.strip()
        if not item or item.lower() in {"na", "n/a"}:
            continue
        # Convert to singular (e.g. "bags" -> "bag")
        words = item.split()
        singular_words = [p.singular_noun(w) if p.singular_noun(w) else w for w in words]
        singular = " ".join(singular_words)
        cleaned.append(singular)

    return cleaned

def clean_location(text):
    # Normalize separators and lower


    items = clean_and_split(text)

    items = re.split(r"[,/;•\n]+", text.lower())
    # items = [clean_item(x) for x in items if x and x.lower() not in {"na", "n/a"}]
    # Remove parentheses or explanatory fragments
    items = [re.sub(r"\(.*?\)", "", x).strip() for x in items]
    
    # Clean extra whitespace and noise
    items = [x.strip() for x in items if x and x != "na" and x != "n/a"]
    # Optionally filter obviously non-location things
    blacklist = {"portable", "edible", "fragile", "sweet", "hot", "cold", "dirty", "clean", "sharp"}
    items = [x for x in items if x not in blacklist and not x.startswith("often ")]
    # print(f"{text} ====-> {sorted(set(items))}")
    return sorted(set(items))  # remove duplicates

def clean_disposition(text):
    if pd.isna(text):
        return []
    canonical_dispositions = {
        # Emotional & Behavioral Traits
        "friendly": "friendly", "loyal": "loyal", "playful": "playful", "affectionate": "affectionate",
        "curious": "curious", "intelligent": "intelligent", "social": "social", "independent": "independent",
        "protective": "protective", "gentle": "gentle", "timid": "timid", "calm": "calm", "bold": "bold",
        "shy": "shy", "skittish": "skittish", "aggressive": "aggressive", "obedient": "obedient",
        "communicative": "communicative", "emotional": "emotional", "complex": "complex",
        "resilient": "resilient", "vulnerable": "vulnerable", "observant": "observant", "docile": "docile",
        "majestic": "majestic", "graceful": "graceful", "alert": "alert", "wary": "wary",
        "quiet": "quiet", "noisy": "noisy", "vocal": "vocal", "entertaining": "entertaining",
        "herbivorous": "herbivorous", "carnivorous": "carnivorous", "omnivorous": "omnivorous",

        # Functional / Physical Properties
        "durable": "durable", "fragile": "fragile", "portable": "portable", "sturdy": "sturdy", "sharp": "sharp",
        "lightweight": "lightweight", "weather-resistant": "weather-resistant", "water-repellent": "water-resistant",
        "water-resistant": "water-resistant", "impact-resistant": "impact-resistant", "corrosion-resistant": "corrosion-resistant",
        "flexible": "flexible", "comfortable": "comfortable", "stable": "stable", "maneuverable": "maneuverable",
        "versatile": "versatile", "resistant": "resistant", "hard-working": "robust", "robust": "robust",
        "reliable": "reliable", "engineered": "engineered", "functional": "functional", "efficient": "efficient",
        "safe": "safe", "fast": "fast", "powerful": "powerful", "agile": "agile", "strong": "strong",
        "rigid": "rigid", "protective": "protective", "breakable": "fragile", "repairable": "repairable",
        "automated": "automated", "stationary": "stationary", "wind-resistant": "weather-resistant",
        "impact-proof": "impact-resistant", "heat-resistant": "heat-resistant", "water-sensitive": "fragile",
        "hygienic": "hygienic", "washable": "washable", "grippy": "grippy", "reusable": "reusable",
        "disposable": "disposable", "biodegradable": "biodegradable", "edible": "edible", "perishable": "perishable",

        # Social / Role-based
        "public": "public", "personal": "personal", "communal": "public", "inviting": "inviting",
        "private": "private", "fashionable": "fashionable", "stylish": "fashionable", "elegant": "elegant",
        "professional": "professional", "utilitarian": "utilitarian", "decorative": "decorative", "functional": "functional",
        "accessory": "fashionable", "entertaining": "entertaining", "essential": "essential", "interactive": "interactive",
        "educational": "educational", "relaxing": "relaxing", "calming": "calming",

        # Misc / Descriptive Modifiers
        "tough": "tough", "soft": "soft", "hard": "hard", "cold-retaining": "insulating", "heat-generating": "heat-resistant",
        "energy-consuming": "energy-consuming", "weatherproof": "weather-resistant", "slightly flexible": "flexible",
        "slightly fragile": "fragile", "scratch-resistant": "resistant", "non-toxic": "non-toxic", "non-slip": "grippy",
        "flimsy": "fragile", "messy": "fragile", "tender": "soft", "juicy": "moist", "sweet": "sweet",
    }
    # Step 1: normalize
    text = text.lower()
    text = re.sub(r"[\.;\-]", ",", text)
    text = re.sub(r"\s+", " ", text)

    # Step 2: extract words before/inside parentheses or separated by commas
    tokens = re.findall(r'\b\w+\b', text)  # basic tokenization
    candidates = set()

    for token in tokens:
        if token in canonical_dispositions:
            candidates.add(canonical_dispositions[token])
    # print(f"{tokens} ====-> {sorted(set(candidates))}")
    return sorted(candidates)

def clean_shape(text):
    if pd.isna(text):
        return []
    canonical_shapes = {
        "cylindrical", "rectangular", "square", "circular", "round", "oval", "oblong",
        "triangular", "conical", "spherical", "tapered", "elongated", "pointed",
        "boxy", "streamlined", "curved", "flat", "compact", "bulky", "sleek", "rounded"
        "ergonomic", "slender", "wide", "thick", "narrow", "tall", "short", "arched",
        "domed", "hourglass", "diamond", "hexagonal", "octagonal", "l-shaped",
        "t-shaped", "u-shaped", "v-shaped", "x-shaped", "teardrop", "aerodynamic",
        "irregular", "humanoid", "bipedal", "quadruped", "animal-shaped", "mitt-like", 
    }
    if pd.isna(text) or text.strip().lower() in {"na", "n/a"}:
        return []

    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()

    tokens = set(text.split())
    matches = set()

    for shape in canonical_shapes:
        # check if shape or partial matches are in the text
        if shape in text:
            matches.add(shape)
        else:
            # try matching tokens to multi-word shapes (e.g., l-shaped)
            if "-" in shape:
                parts = shape.split("-")
                if all(p in tokens for p in parts):
                    matches.add(shape)
    print(f"\n{text}\n=> {sorted(matches)}\n")
    return sorted(matches)

def clean_materials(text):
    if pd.isna(text):
        return []

    canonical_map = {
        # Spelling variants and synonyms
        "aluminium": "aluminum", "fibreglass": "fiberglass", "carbonfiber": "carbon fiber",
        "carbon fibre": "carbon fiber", "styrofoam": "foam", "synthetic leather": "leather",
        "fake leather": "leather", "faux leather": "leather", "pleather": "leather",
        "timber": "wood", "hardwood": "wood", "softwood": "wood", "cloth": "fabric",
        "textile": "fabric", "woven": "fabric", "polyethylene": "plastic", "polystyrene": "plastic",
        "polypropylene": "plastic", "polycarbonate": "plastic", "acryl": "acrylic",
        "acrylate": "acrylic", "porcelaine": "porcelain", "stoneware": "ceramic",
        "china": "ceramic", "silicone rubber": "silicone", "natural rubber": "rubber",
        "latex rubber": "latex", "rubberized": "rubber", "card board": "cardboard",
        "muscular tissue": "muscle", "skin tissue": "skin", "hair strands": "hair",
        "animal skin": "skin", "painted surface": "paint", "ply wood": "plywood",
        "pressed wood": "plywood", "wood composite": "plywood", "particle board": "plywood",
        "hard board": "plywood", "cork board": "cork", "compressed paper": "cardboard",
        "laminated wood": "plywood", "vellum": "paper", "vellum paper": "paper", "sponge": "foam",
        "spongy": "foam", "resin composite": "resin", "clear plastic": "plastic",
        "soft plastic": "plastic", "hard plastic": "plastic", "heated glass": "glass",
        "tempered glass": "glass", "bulletproof glass": "glass", "shatterproof glass": "glass",
        "wire glass": "glass", "safety glass": "glass", "glass panel": "glass", "flesh": "muscle",
        "bio-material": "biomaterial", "biomaterial": "biomaterial", "tissue": "biomaterial",
        "fur": "hair", "hairy": "hair", "animal hair": "hair", "animal wool": "wool",
        "human hair": "hair", "copper wire": "copper", "iron rod": "iron", "metal wire": "metal",
        "metallic": "metal", "mirror": "glass", "mirrored surface": "glass", "crystal": "glass",
        "crystalline": "glass", "glasslike": "glass", "marble stone": "marble",
        "granite stone": "granite", "natural stone": "stone", "concrete mix": "concrete",
        "cement mix": "cement", "stone dust": "stone", "waxed": "wax", "silicon": "silicone",
        "epoxy": "resin", "epoxy resin": "resin", "enamel": "paint", "oil paint": "paint",
        "latex paint": "paint", "acrylic paint": "paint", "water": "liquid", "oil": "liquid",
        "air": "gas", "gasoline": "liquid", "diesel": "liquid", "gelatin": "gel", "jelly": "gel",
        "goo": "gel", "mud": "clay", "sludge": "clay", "sandstone": "stone", "rock": "stone",
        "crushed rock": "stone", "gravel": "stone", "grit": "stone", "ash": "powder",
        "powdered": "powder", "dust": "powder", "cotton" : "cotton", "stainless steel": "stainless steel",
        "crystal" : "crystal", "linen" : "linen", "down": "down", "organic" : "organic", "flour" : "flour",
        "egg" : "egg", "milk": "milk", "butter": "butter", "sugar":"sugar", "cheese":"cheese", "dough":"dough",
        "food": "organic", "meat": "organic", "natural" : "organic"
    }

    text = text.lower()
    text = re.sub(r"[;\.\-/]", ",", text)
    text = re.sub(r"\s+", " ", text)

    found = set()
    used_text = text

    # Phrase-level replacement
    for phrase, canonical in canonical_map.items():
        if phrase in used_text:
            found.add(canonical)
            # Optional: remove phrase from text to prevent double match
            used_text = used_text.replace(phrase, "")

    # Now also extract any remaining single words (e.g., "metal, plastic")
    tokens = re.findall(r'\b\w+\b', text)
    values = list(canonical_map.values())
    for token in tokens:
        if token in values:
            found.add(canonical_map.get(token, token))  # keep as-is if not mapped

    print(f"\n{text}\n=> {sorted(found)}\n")
    return sorted(found)


def clean_function(text):
    if pd.isna(text) or text.strip().lower() in {"na", "n/a"}:
        return []
    canonical_functions = {
        "eat", "drink", "cook", "store food", "reheat", "cut", "pierce", "scoop",
        "communicate", "learn", "entertain", "decorate", "carry", "transport",
        "commute", "travel", "signal", "regulate traffic", "sleep", "sit", "rest",
        "socialize", "play", "educate", "navigate", "measure time", "clean", "wash",
        "observe", "provide warmth", "store items", "protect", "gather", "fly", 
        "swim", "glide", "support", "exercise", "train", "serve food", "walk",
        "think", "create", "build", "guide", "guard", "rescue", "breed", "graze",
        "hunt", "forage", "hibernate", "lay eggs", "sing", "pollinate", "disperse seeds",
        "provide milk", "provide wool", "provide meat", "produce leather", 
        "provide companionship", "display time", "broadcast", "browse internet",
        "input text", "click", "control devices", "blow air", "dry hair", "clean teeth"
    }

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    matches = set()

    # Try direct match
    for fn in canonical_functions:
        if fn in text:
            matches.add(fn)

    # Optionally use fuzzy match for rare edge cases
    tokens = set(text.split())
    for token in tokens:
        close = get_close_matches(token, canonical_functions, n=1, cutoff=0.92)
        if close:
            matches.add(close[0])
    print(f"{tokens} ====-> {sorted(set(matches))}")
    return sorted(matches)

def clean_pattern(text):
    if pd.isna(text):
        return []
    canonical_patterns = {
        "plain", "solid color", "striped", "checkered", "plaid", "polka-dotted", "dotted", "spotted", "patchy", "mottled",
        "camouflage", "graphic", "floral", "geometric", "abstract", "textured", "ribbed", "paneled", "grooved", "engraved",
        "logo", "branded", "printed", "illustrated", "lettered", "written", "swirled", "marbled", "sprinkled", "tiled",
        "gradient", "decorative", "laced", "stitched", "quilted", "monogrammed", "veined", "variegated", "fractal",
        "tartan", "animal print", "cartoon", "striped (racing)", "engraved", "damascus", "paneled", "paneled", "grid", "glossy", "speckled",
        "random", "symmetrical", "florets", "solid"
    }


    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    matches = set()

    for pattern in canonical_patterns:
        if pattern in text:
            matches.add(pattern)
    print(f"\n{text}\n=> {sorted(matches)}\n")
    return sorted(matches)

def clean_colours(text):
    if pd.isna(text):
        return []
    canonical_colors = {
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "orange": "orange",
        "purple": "purple",
        "pink": "pink",
        "brown": "brown",
        "black": "black",
        "white": "white",
        "gray": "gray",
        "grey": "gray",  # UK/US spelling
        "silver": "silver",
        "gold": "gold",
        "beige": "beige",
        "cream": "cream",
        "tan": "tan",
        "navy": "navy",
        "maroon": "maroon",
        "burgundy": "burgundy",
        "bronze": "bronze",
        "charcoal": "gray",
        "clear": "transparent",
        "transparent": "transparent",
        "colorless": "transparent",
        "multicolor": "multicolored",
        "multi-colored": "multicolored",
        "multi": "multicolored",
        "rainbow": "multicolored",
        "patterned": "patterned",
        "natural wood": "brown",
        "wood": "brown",
        "metallic": "silver",
        "stainless steel": "silver",
        "ceramic": "white",
        "rose gold": "gold",
        "off-white": "white",
        "dark blue": "blue",
        "light blue": "blue",
        "light gray": "gray",
        "dark gray": "gray",
        "dark green": "green",
        "light green": "green",
        "reddish-brown": "brown",
        "golden": "gold",
        "blond": "yellow",
        "ginger": "orange",
        "chrome": "silver",
        "camo": "multicolored",
        "varied": "multicolored",
        "various": "multicolored",
        "custom": "multicolored",
    }
    # Step 1: normalize
    text = text.lower()
    text = re.sub(r"[\.;\-]", ",", text)
    text = re.sub(r"\s+", " ", text)

    # Step 2: extract words before/inside parentheses or separated by commas
    tokens = re.findall(r'\b\w+\b', text)  # basic tokenization
    candidates = set()

    for token in tokens:
        if token in canonical_colors:
            candidates.add(canonical_colors[token])
    if not candidates:
        print(f"No candidates found for {tokens}")
    # else:
    #     print(f"[OK] {tokens} ==> {candidates}")
    return sorted(candidates)

def is_adjective(word):
    """Check if a word or its hyphenated parts are adjectives."""
    STOPWORDS = {'some', 'on', 'in', 'and', 'or', 'of', 'the', 'a', 'an', 'to', 'with', 'for'}
    if word in STOPWORDS:
        return False
    if wn.synsets(word, pos=wn.ADJ):
        return True
    if '-' in word:
        parts = word.split('-')
        return all(wn.synsets(part, pos=wn.ADJ) for part in parts if part not in STOPWORDS)
    return False

def extract_adjectives(text):
    if pd.isna(text):
        return []
    STOPWORDS = {'some', 'on', 'in', 'and', 'or', 'of', 'the', 'a', 'an', 'to', 'with', 'for'}
    # Normalize text
    text = text.lower()
    text = re.sub(r"[.;_/]", ",", text)  # keep hyphens
    text = re.sub(r"\s+", " ", text)

    tokens = re.split(r"[,\n]", text)
    tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in {'na', 'n/a'}]

    adjectives = []
    for token in tokens:
        words = token.split()

        for i, word in enumerate(words):
            if word == 'not' and i + 1 < len(words):
                # Skip the adjective following 'not'
                continue
            if i > 0 and words[i - 1] == 'not':
                # Skip if preceded by 'not'
                continue
            if is_adjective(word):
                adjectives.append(word)

    print(f"{tokens} ====-> {sorted(set(adjectives))}")
    return sorted(set(adjectives))

def clean_categorical_columns(df):
    
    df["Answer.texture_cleaned"] = df["Answer.texture"].apply(clean_texture)
    df["Answer.colour_cleaned"] = df["Answer.colour"].apply(clean_colours)
    df["Answer.location_cleaned"] = df["Answer.location"].apply(clean_location)
    df["Answer.disposition_cleaned"] = df["Answer.disposition"].apply(extract_adjectives)
    df["Answer.material_cleaned"] = df["Answer.material"].apply(clean_materials)
    # df["Answer.function_cleaned"] = df["Answer.function"].apply(clean_function) # SKIPPED
    df["Answer.shape_cleaned"] = df["Answer.shape"].apply(clean_shape)
    df["Answer.pattern_cleaned"] = df["Answer.pattern"].apply(clean_pattern)

    # cat_columns = ['Answer.'+col for col in ['disposition', 'location', 'material', 'function', 'texture', 'pattern', 'shape']]
    # """Apply tokenization to selected categorical columns"""
    # for col in cat_columns:
    #     if col in df.columns:
    #         df[col + "_cleaned"] = df[col].apply(tokenize_clean)

    return df

def keep_relevant_columns(df):
    # Always keep these first if they exist
    first_columns = ["WorkerId"]
    relevant = [col for col in df.columns if col.startswith("Input.") or col.startswith("Answer.")]
    all_cols = first_columns + [col for col in relevant if col not in first_columns]
    return df[all_cols].copy()

def collapse_unit_columns(df, unit_columns, new_col_name="unit"):

    # Extract unit labels from column names
    unit_labels = [col.split('.')[-1] for col in unit_columns]

    # Apply row-wise logic
    def get_unit(row):

        for col in unit_columns:
            if row[col] == 'true':
                return col.split('.')[-1]
        return 'na'

    df[new_col_name] = df[unit_columns].apply(get_unit, axis=1)
    return df

def aggregate_measurement_unit_columns(df):
    columns_to_aggregate = {}

    for column in df.columns:
        if 'unit_' in column:
            stem = column.split('_')[0]

            # Check if column has at least one truthy value (bool, "True", or 1)
            if df[column].astype(str).str.lower().isin(["true", "1"]).any():
                columns_to_aggregate.setdefault(stem, []).append(column)

    for dim, cols in columns_to_aggregate.items():
        df = collapse_unit_columns(df, cols, new_col_name=dim + '_unit')

    return df

def smart_parse_float(value):
    original = value
    if pd.isna(value):
        return 'na'
    value = str(value).strip().lower()

    if value in ["na", "n/a", "none", ""]:
        return 'na'

    # Normalize dashes to standard hyphen, remove tilde/approx symbols
    value = value.replace("–", "-").replace("—", "-")
    value = value.replace("~", "")

    # Remove currency or unit symbols before parsing
    value = re.sub(r"[$€£¥]", "", value)

    # Fix comma-separated numbers like "13,000" → "13000"
    value = re.sub(r'(?<=\d),(?=\d)', '', value)

    if "mean (average):" in value:
        try:
            after_mean = value.split("mean (average):", 1)[1]
            num_match = re.search(r"[-+]?\d*\.\d+|\d+", after_mean)
            if num_match:
                parsed = float(num_match.group())
                # print(f"[MEAN AVG] '{original}' → {parsed}")
                return parsed
        except Exception:
            pass

    try:
        return float(value)
    except ValueError:
        # Extract numbers
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)

        if len(numbers) == 1:
            parsed = float(numbers[0])
            # print(f"[OK] Extracted: '{original}' → {parsed}")
            return parsed

        elif len(numbers) > 1:
            # Heuristic 1: hyphen range
            if '-' in value and re.search(r'\d+(\.\d+)?\s*-\s*\d+(\.\d+)?', value):
                try:
                    nums = [float(n) for n in numbers[:2]]
                    avg = sum(nums) / 2
                    # print(f"[RANGE AVG] '{original}' → avg({nums[0]}, {nums[1]}) = {avg}")
                    return avg
                except Exception:
                    pass
            # "X to Y" pattern
            if re.search(r'\d+(\.\d+)?\s+to\s+\d+(\.\d+)?', value):
                try:
                    nums = [float(n) for n in numbers[:2]]
                    avg = sum(nums) / 2
                    # print(f"[TO AVG] '{original}' → avg({nums[0]}, {nums[1]}) = {avg}")
                    return avg
                except Exception:
                    pass

            # Heuristic 2: before parentheses
            if '(' in value:
                try:
                    before_paren = value.split('(')[0]
                    nums_before = re.findall(r"[-+]?\d*\.\d+|\d+", before_paren)
                    if len(nums_before) >= 2:
                        avg = (float(nums_before[0]) + float(nums_before[1])) / 2
                        # print(f"[PAREN AVG] '{original}' → avg({nums_before[0]}, {nums_before[1]}) = {avg}")
                        return avg
                    elif len(nums_before) == 1:
                        parsed = float(nums_before[0])
                        # print(f"[PAREN FIRST] '{original}' → {parsed}")
                        return parsed
                except Exception:
                    pass

            print(f"[AMBIGUOUS] '{original}' → multiple numbers found: {numbers}")
            return None

        else:
            # print(f"[FAIL] Could not parse: '{original}' → None")
            return 'na'



def clean_mean_values(df):
    for col in df.columns:
        if '_mean' in col:
            df[col] = df[col].apply(smart_parse_float)
    return df

def clean_range_values(df):
    for col in df.columns:
        if '_range' in col:
            df[col] = df[col].apply(smart_parse_float)
    return df

def main(input_file, output_file):
    df = pd.read_csv(input_file, quotechar='"', skipinitialspace=True, dtype=str)
    df = keep_relevant_columns(df)
    df = aggregate_measurement_unit_columns(df)
    df = clean_mean_values(df)
    df = clean_range_values(df)
    df = clean_categorical_columns(df)
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to: {output_file}")

if __name__ == "__main__":

    main("data/ground_truth/1_batch_raw_results.csv", "cleaned_results.csv")
