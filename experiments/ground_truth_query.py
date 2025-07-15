from SPARQLWrapper import SPARQLWrapper, JSON
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')


def find_predicates_by_label(label_keyword):
    logging.info(f"Querying DBpedia for predicates matching label: '{label_keyword}'")
    
    endpoint = SPARQLWrapper("http://dbpedia.org/sparql")
    keyword = label_keyword.lower().replace('"', '')

    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?property ?label WHERE {{
      ?property a rdf:Property ;
                rdfs:label ?label .
      FILTER (lang(?label) = 'en')
      FILTER CONTAINS(LCASE(STR(?label)), "{keyword}")
    }}
    LIMIT 1000
    """

    endpoint.setQuery(query)
    endpoint.setReturnFormat(JSON)
    endpoint.setTimeout(10)

    try:
        results = endpoint.query().convert()
        bindings = results["results"]["bindings"]
        if not bindings:
            print(f"❌ No predicates found for keyword '{label_keyword}'.")
            return []
        matches = []
        for result in bindings:
            uri = result["property"]["value"]
            label = result["label"]["value"]
            matches.append(uri)
            print(f"🔗 {uri}\n   📛 {label}\n")
        print(f"\n🔍 Found {len(bindings)} predicates for keyword '{label_keyword}':\n")
        return matches
    
    except Exception as e:
        logging.error(f"SPARQL query failed: {e}")
        return []


def fetch_example_triples(predicate_uri, limit=10):
    logging.info(f"Fetching example triples for property: {predicate_uri}")
    endpoint = SPARQLWrapper("http://dbpedia.org/sparql")

    query = f"""
    SELECT DISTINCT ?s ?o WHERE {{
      ?s <{predicate_uri}> ?o .
    }} LIMIT {limit}
    """

    endpoint.setQuery(query)
    endpoint.setReturnFormat(JSON)
    endpoint.setTimeout(10)

    try:
        results = endpoint.query().convert()
        bindings = results["results"]["bindings"]
        if not bindings:
            print(f"❌ No triples found for predicate {predicate_uri}.")
            return

        print(f"\n📊 Example triples for predicate: {predicate_uri}\n")
        for result in bindings:
            subject = result["s"]["value"]
            obj = result["o"]["value"]
            print(f"  ▶ {subject} → {obj}")
    except Exception as e:
        logging.error(f"Failed to fetch triples: {e}")


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Mode: search for matching predicates
        keyword = sys.argv[1]
        find_predicates_by_label(keyword)

    elif len(sys.argv) == 3 and sys.argv[1] == "--triples":
        # Mode: fetch example triples for a given predicate URI
        uri = sys.argv[2]
        fetch_example_triples(uri)

    else:
        print("Usage:")
        print("  python script.py <label_keyword>           # Search predicates by label")
        print("  python script.py --triples <predicate_uri>  # Get example triples")
