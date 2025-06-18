# Constructing Commonsense Knowledge Graphs with Large Language Models

There are two inputs to the system. One is the concepts themselves that need to be described by the set of quality dimensions, and the other input is the quality dimensions describing the entity.
To describe the entity, we first have to determine which set of domains are relevant to the entity at hand. This is given by the context vector, which acts as a binary filter and describes the quality dimensions that are relevant to the concept. For example, we do not care about the speed of an apple, or the colour of a shape.
There are two main cases to consider here. The first case is when the quality dimension does not apply to the concept (e.g colour of a shape), and when the value can take any form, as it is not a defining feature of the concept at hand (e.g. the colour of a mug).
We present here an initial analysis of the commonsense knowledge extraction abilities of the agents using LLMs, and using the 80 concepts described in the MS COCO dataset.
Then we extend the knowledge extraction pipeline to include the 1000 concepts in imagenet-1k dataset.
