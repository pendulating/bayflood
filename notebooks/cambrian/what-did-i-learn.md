In this subproject, we took the recently (June 24) released multimodal LLM model, Cambrian, and investigated its performance on our street flooding task. We used the 13B-variant of the model, which sits between the 8B-sized and 34-B sized weights. We ran out of RAM when using the 34-B sized weights.

We test two different capabilities with two different types of questions: 
(1) We test how good Cambrian-13B is at assessing whether a dashcam image shows flooding or not. 
    - "Does this image show a flooded street?"
    - "Does this image show more than a foot of standing water?"
    - "Is the street in this image flooded?"
    - "Could a car drive through the water in this image?"
(2) We test how good Cambrian-13B is at assessing whether an image even shows enough to classify flooding. 
    - "Does this image show a visible street?"
    - "Is there any visible street in this image?"
    - "Is the view from windshield in this image too obstructed?"
