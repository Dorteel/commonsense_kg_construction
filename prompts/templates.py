system_prompt = "You are a commonsense knowledge engineer. Your task is to provide accurate commonsense knowledge. Return **ONLY** valid JSON."

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

def template_classical_avg(concept, description, domain, dimension, measurement=None):
    if measurement:
        prompt=f"""Given the {concept} (which is a {description}),
        provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')} measured in {measurement.replace('_', ' ')}
        Return only JSON with key:
            - {dimension} (float)
        """
    else:
        prompt=f"""Given the {concept} (which is a {description}),
        provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')}
        Return only JSON with key:
            - {dimension} (string)
        """
    return prompt



def template_chain_of_thought_avg(concept, description, domain, dimension, measurement=None):
    pass

def template_self_verification_avg(concept, description, domain, dimension, measurement=None, prev_answer=None):
    pass

def instructor_classical_avg(concept, description, domain, dimension, measurement=None):
    if measurement:
        prompt=f"""Given the {concept} (which is a {description}),
        provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')} measured in {measurement.replace('_', ' ')}
        Return only the value for {dimension} in (float)
        """
    else:
        prompt=f"""Given the {concept} (which is a {description}),
        provide the average {domain.replace('_', ' ')} as in {dimension.replace('_', ' ')}
        Return only the value for {dimension} in (string)
        """
    return prompt