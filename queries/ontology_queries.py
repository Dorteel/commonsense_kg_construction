"""Reusable SPARQL queries for the emotion ontology."""

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
