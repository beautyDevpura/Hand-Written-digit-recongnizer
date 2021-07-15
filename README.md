# Hand-Written-digit-recongnizer

For educational purpose and using NO Deep learining model for prediction, I used the very famous - MNIST dataset
I have created three different files as MNIST dataset has 70000 elements with 785 columns(784 dimensions = 28*28) and 1 target(labels) column.
The labels are from 0 to 9, so we have 10 classes here. 
One more point I want to add is, I have used MNIST data from fetch_openml library for first two files and used local mnist Data set file for third file which has lesser pixel images than available in fetch_openml.

#Visualization

![Screen Shot 2021-07-13 at 11 01 00 AM](https://user-images.githubusercontent.com/82862957/125855776-31fd9f84-e3b6-4113-9e72-73fd72b3b2e1.png)

This image is of Data of one image with its 784 dimension in Flatten form.

![Screen Shot 2021-07-13 at 11 19 28 AM](https://user-images.githubusercontent.com/82862957/125855921-bc888a2b-ece3-42dd-a215-cce8a0ecb4b5.png)

This is in a form of Dataframe to visualize combination Zeros and other numebers forming a digit - 8 in 28*28 pixels form

![download](https://user-images.githubusercontent.com/82862957/125856306-19573037-9e66-4300-be87-ba173726d55f.png)

One of the Data element which has label 8 in 28*28 pixel format. Zeros form color black where other numbers form grays and white

![image1](https://user-images.githubusercontent.com/82862957/125856490-f01d683c-1834-42bb-a275-c9f82576aaca.png)

when image is not converted to gray 

![download (1)](https://user-images.githubusercontent.com/82862957/125856702-3c8456c4-f217-4a5a-a958-908df903717b.png)

Other pixel images from Dataset

#model_PKL:
In this file, mnist Dataset has been recalled for actual purpose of model run
I have used KNN Classifier model which gave around 93% of accuracy and duped the model using PICKLE library to call it in another file

#StreamlitCanvas:

Created Stremlit canvas using stremlit_drawable_canvas library
I have used local mnist data and give filepath of my local mnist loaction. 

The reason behind to used local mnist data is whenever there is changes made to canvas, whole codefile will be re-executed and using local mnist data make this process faster.

I have used Pickle to load the model saved in model_PKL file, resize the image to 28*28 taken from Canvas Drawing which must be ny number from 0-9.
And last predict button will predic the digit drawn in Canvas.

![Screen Shot 2021-07-15 at 2 26 50 PM](https://user-images.githubusercontent.com/82862957/125860271-bec9e8ca-3d81-4996-88ce-5afc6a581651.png)

Streamlit Canvas on local host 

Link of Drawable Canvas and model predicting Digit from 0-9
https://github.com/beautyDevpura/Hand-Written-digit-recongnizer/blob/main/StreamlitCanvas_Draw_predict.webm
