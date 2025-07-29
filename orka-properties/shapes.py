import requests
import yaml
import os

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT DISTINCT ?shape ?shapeLabel ?altLabel WHERE {
  ?thing wdt:P1419 ?shape.
  ?shape rdfs:label ?shapeLabel.
  FILTER(LANG(?shapeLabel) = "en")
  OPTIONAL {
    ?shape skos:altLabel ?altLabel.
    FILTER(LANG(?altLabel) = "en")
  }
}
"""

def query_wikidata_shapes():
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "ShapeAliasBot/2.0 (now with canonical allegiance)"
    }
    response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={'query': QUERY}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Wikidata query failed: {response.status_code}")
    return response.json()

def process_shape_aliases(data):
    alias_map = {}
    for result in data['results']['bindings']:
        canonical_label = result['shapeLabel']['value'].strip().lower()
        alias = result.get('altLabel', {}).get('value')
        
        # Always include canonical label as its own alias
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
    print(f"Saved {len(data)} shape alias mappings to {filename}")

if __name__ == "__main__":
    data = query_wikidata_shapes()
    alias_map = process_shape_aliases(data)
    save_to_yaml(alias_map, "shape_alias_map.yaml")
