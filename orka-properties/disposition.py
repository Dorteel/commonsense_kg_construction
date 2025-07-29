    
import yaml
import os

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

def save_to_yaml(data, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=True, allow_unicode=True)
    print(f"Saved {len(data)} labels to {filename}")


if __name__ == "__main__":
    save_to_yaml(canonical_dispositions, "dispositions.yaml")