# plant_leaf_health_detector
My first CNN (and NN in general). Still lots of refining for me to do. This is based on the caveman prompt that I was given. Where a DNN model is made to optimize the life of cavemen assuming they have access to electricity and a computer but not much besides that. 
This  just used a Kaggle dataset (https://www.kaggle.com/soumiknafiul/plantvillage-dataset-labeled) and applies CNNs to it. The idea is that the caveman's next step is to stop searching for food all day, and get started with agriculture. So that he can have more time to work on things like calculus and stone tools and other evolution stuff.

BUT, the only way to do that is to grow healthy plants and be smart about growing them. This model does that. It uses the dataset to learn (from a small set of 5 vegetables and fruits) to classify which  are in bad shape and which are healthy. So it's not just a simple dog/cat classifier but classifies for  like 19 labels (i think label is the right word).

The model is not very accurate (I got 85% val_acc on average) as I was having difficulties with overfitting. Also, there are more things I want to understand in addition to combating overfitting, like how to use different functions and all the parameters in the Keras methods to have a better model. Even the code that predicts using the model is bad, but I'm pretty sure I am doing something wrong there. The cool thing is, I have some tomato plants and I can test those out with this model. But the images that I was predicting it with were either poorly selected from the internet or not close-ups. 


i have a good groundwork for this little project to improve and learn more on. This was a good assignment to quickly apply some of the theory that I have been collecting on CNNs but I will defintely refine this more. 
