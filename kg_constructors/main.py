
from kg_constructors.pipeline import ConstructionPipeline

# Load the concept

# Load the properties

# Create a template

# Load the model and run the prompt

# Extract info from the input

# Add output to the ontology

if __name__ == "__main__":
    pipeline = ConstructionPipeline('apple')
    pipeline.run()
    print("Knowledge graph construction completed successfully.")