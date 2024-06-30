# Urban Street Flooding Project - Summer 2024 
This project started in the fall of 2023, and has been mired in various different directions. In this repo, we take the working insights we learned during from existing results and start with a clean slate, allowing for better reproducibility and an easier-to-follow project narrative. 


## Objective 
The objective of this project is to take over 900 thousand dashcam images from a day of severe flooding in New York City, and train a high-performance, generalizable flooding classifier with minimal human labeling (as is typical in the supervised learning paradigm). 


## Experiment 
We craft an initial benchmark set of 1000 flooded images and 5000 non-flooded images using a combination of CLIP, GPT4-V (preview), and human labeling of high-confidence images. We aim to release this dataset as a useful benchmark for flood classification in urban environments. We additionally offer supplemental datasets of images from two other days of severe flooding in late 2023 and early 2024. 

We choose not to rely on CLIP solely, as its performance was lower than that of CLIP paired with GPT4-V as a second pass filter. Moreover, we choose not to use GPT4-V in our analysis, as the weights are not open-sourced and so reproducibility is not guaranteed. 

With this benchmark set, we create a prompt engineering experiment for a visual question answering task, using the recently released Cambrian-1M multimodal large language model (LLM).

Once we achieve high-enough classification accuracy from a good prompt, we run inference on the entire set of 900k plus images, and manually verify a random sample of 0.1% images to estimate precision & recall on the overall set. 

We take the inference results from the entire set and use detected positives as an *analysis set*. Downstream analyses to perform start with looking at extended coverage from the medium of dashcam, relative to FloodNet sensors, 311 reports, and predicted stormwater flooding maps created by the New York City Department of Environmental Protection. Then, we move to looking at biases in 311 flooding reports, using detected positives as a first-in-literature visual ground truth. 

## Model Training 
We train an image classification model on the *analysis set*, crafting a 60-20-20 training-val-test split. We baseline this model against the aforementioned *benchmark set* of around 6000 images, which has the advantage of being entirely human-verified. 

This model is useful for downstream applications, including a real-time deployment on vehicle dashcams for flood detection. 
