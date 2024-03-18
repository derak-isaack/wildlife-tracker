## Wildlife tracker

Wildlife are a very important factor in our ecosystem as they help balance natural ecosystems. They also offer revenues for many countries through tourist attraction sites, game drives and conservation hubs. One problem evident is how sometimes the ever changing climate sometimes drives them out from the forests thereby resulting into human-wildlife conflict. This often results into destruction of property and loss of human life in some instances. 

## Training

Training deep learning models for wildlife tracking on CPUs can be time-consuming. Utilizing `GPUs`, such as those provided by `Google Colab`, is essential for efficient training. Models will be trained on GPUs and later saved for deployment using torch on CPUs. This documentation[https://pytorch.org/tutorials/beginner/saving_loading_models.html] comes in handy to understand loading of `GPU` trained models on the `CPU`.

## Deployment 

Streamlit or Neural magic will be the tools for deployment. The later is however reccommended because of its scalability properties and efficiency in handling large data. As this is a deep learning model, deploying it uisng **Neural magic** will offer the efficiency required. 