import requests
import yaml
import os

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?color ?colorLabel ?altLabel (COUNT(?s) AS ?usage) WHERE {
  ?s wdt:P462 ?color.
  ?color rdfs:label ?colorLabel.
  FILTER(LANG(?colorLabel) = "en")
  OPTIONAL {
    ?color skos:altLabel ?altLabel.
    FILTER(LANG(?altLabel) = "en")
  }
}
GROUP BY ?color ?colorLabel ?altLabel
ORDER BY DESC(?usage)
LIMIT 100
"""

def query_wikidata_colors():
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "ColorAliasBot/3.0 (the dictator of chromatic standardization)"
    }
    response = requests.post(WIKIDATA_SPARQL_ENDPOINT, data={'query': QUERY}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Wikidata query failed: {response.status_code}")
    return response.json()

def process_color_aliases(data):
    alias_map = {}
    for result in data['results']['bindings']:
        canonical_label = result['colorLabel']['value'].strip().lower()
        alias = result.get('altLabel', {}).get('value')

        alias_map[canonical_label] = canonical_label
        if alias:
            alias_lower = alias.strip().lower()
            alias_map[alias_lower] = canonical_label

    return dict(sorted(alias_map.items()))

def save_to_yaml(data, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=True, allow_unicode=True)
    print(f"Saved {len(data)} color alias mappings to {filename}")

if __name__ == "__main__":
    data = query_wikidata_colors()
    alias_map = process_color_aliases(data)
    save_to_yaml(alias_map, "color_alias_map.yaml")
