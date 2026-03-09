"""Reusable SPARQL queries for the emotion ontology."""

DEFINITION_SOURCE_TO_PROPERTY = {
    "apa": "emo:hasDefinition_APA",
    "cambridge": "emo:hasDefinition_Cambridge",
    "sentiwordnet": "emo:hasDefinition_SentiWordNet",
    "generic": "emo:hasDefinition",
}


def emotion_concepts_query_for_source(definition_source: str) -> str:
    """Build concept query for one definition source.

    Supported values:
    - apa
    - cambridge
    - sentiwordnet
    - generic
    - merged (fallback chain: APA -> Cambridge -> SentiWordNet -> generic)
    """
    source_key = str(definition_source).strip().lower()
    if source_key == "merged":
        return EMOTION_CONCEPTS_QUERY

    if source_key not in DEFINITION_SOURCE_TO_PROPERTY:
        allowed = sorted(list(DEFINITION_SOURCE_TO_PROPERTY.keys()) + ["merged"])
        raise ValueError(f"Unsupported definition source: {definition_source}. Allowed: {allowed}")

    definition_property = DEFINITION_SOURCE_TO_PROPERTY[source_key]
    return f"""
PREFIX emo: <http://example.org/emotions.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?concept ?name ?definition
WHERE {{
  ?concept a emo:Emotion ;
           rdfs:label ?name .
  OPTIONAL {{ ?concept {definition_property} ?definition_raw . }}
  BIND(COALESCE(STR(?definition_raw), "") AS ?definition)
}}
ORDER BY LCASE(STR(?name))
"""


EMOTION_CONCEPTS_QUERY = """
PREFIX emo: <http://example.org/emotions.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?concept ?name ?definition
WHERE {
  ?concept a emo:Emotion ;
           rdfs:label ?name .

  OPTIONAL { ?concept emo:hasDefinition_APA ?definition_apa . }
  OPTIONAL { ?concept emo:hasDefinition_Cambridge ?definition_cambridge . }
  OPTIONAL { ?concept emo:hasDefinition_SentiWordNet ?definition_senti . }
  OPTIONAL { ?concept emo:hasDefinition ?definition_generic . }

  BIND(COALESCE(?definition_apa, ?definition_cambridge, ?definition_senti, ?definition_generic, "") AS ?definition)
}
ORDER BY LCASE(STR(?name))
"""

EMOTION_DOMAINS_QUERY = """
PREFIX conc: <http://example.org/conceptual_spaces.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?domain ?name ?description ?type
WHERE {
  ?space conc:hasDomain ?domain .
  OPTIONAL { ?domain rdfs:label ?name . }

  BIND(CONCAT("Emotion domain/model: ", COALESCE(STR(?name), STR(?domain))) AS ?description)
  BIND("categorical" AS ?type)
}
ORDER BY LCASE(STR(?name))
"""

EMOTION_DIMENSIONS_QUERY = """
PREFIX conc: <http://example.org/conceptual_spaces.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX emo: <http://example.org/emotions.owl#>

SELECT DISTINCT ?domain ?name ?description ?type ?range ?range_min ?range_max ?domain_model
WHERE {
  ?space conc:hasDomain ?domain_model .
  ?domain_model conc:hasQualityDimension ?dimension .

  OPTIONAL { ?domain_model rdfs:label ?domain_label . }
  OPTIONAL { ?dimension rdfs:label ?name . }
  OPTIONAL { ?dimension emo:hasDefinition ?definition . }
  OPTIONAL { ?dimension conc:hasRange ?range . }
  OPTIONAL { ?dimension emo:hasRangeMin ?range_min . }
  OPTIONAL { ?dimension emo:hasRangeMax ?range_max . }

  BIND(?dimension AS ?domain)
  BIND(COALESCE(STR(?definition), "No definition provided.") AS ?description)
  BIND("emotion" AS ?type)
}
ORDER BY LCASE(STR(?name))
"""
