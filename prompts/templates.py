def template_measurement(concept, description, domain, dimension, measurement):

    return f"""Given the {concept} (which is a {description}),
    provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')} measured in {measurement.replace('_', ' ')}
    Return only JSON with key:
        - {dimension} (float)
    """

def template_categorical(concept, description, domain, dimension):
    
    return f"""Given the {concept} (which is a {description}),
    provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')}
    Return only JSON with key:
        - {dimension} (string)
    """