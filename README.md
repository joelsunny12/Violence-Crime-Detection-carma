# carma

### Problem Statement

In video surveillance, to critically assure public safety, hundreds and thousands of surveillance cameras are deployed within cities, but it is almost impossible to manually monitor all cameras to keep an eye on violent activities. Rather, there is a significant requirement for developing automated video surveillance systems to automatically track and monitor such activities.

### Models used
1. LRCN (CNN + LSTM) [Long-term Recurrent Convolutional Network]
2. ConvLSTM [Convolutional Long short term memory]

<br>
ConvLSTM is a variant of LSTM (Long Short-Term Memory) containing a convolution operation inside the LSTM cell. Both the models are a special kind of RNN, capable of learning long-term dependencies.

ConvLSTM replaces matrix multiplication with convolution operation at each gate in the LSTM cell. By doing so, it captures underlying spatial features by convolution operations in multiple-dimensional data.

The main difference between ConvLSTM and LSTM is the number of input dimensions. As LSTM input data is one-dimensional, it is not suitable for spatial sequence data such as video, satellite, radar image data set. ConvLSTM is designed for 3-D data as its input.

A CNN-LSTM is an integration of a CNN (Convolutional layers) with an LSTM. First, the CNN part of the model process the data and one-dimensional result feed an LSTM model. 


<br>

**Flowchart of carma**

![image](https://user-images.githubusercontent.com/73103188/176588336-741b4b77-134a-4f50-9797-5d3ad23ba17f.png)

 **Demo Video of Carma**

https://user-images.githubusercontent.com/73103188/176587208-aa1dac0a-ef95-4404-bc74-f62bd88292a2.mp4

 **Explanation**
 
 The video which has to be classified will be fed into the site of carma. The LRCN model is run on it and the output will be downloaded, which is the labelled version of the input video, being classified into fight/no-fight.
 
 **Accuracy Achieved**
 73.33% using LRCN
 64.57% using ConvLSTM
 
 **Future Enhancements**

• We will connect our model to a cctv system for realtime violence detection.\
• We are going to build a notification system that will send a ping to the app whenever it detection any sort of crime.\
• The location should also get updated on the map without any delay so that the police can get to know where the crime is happening.\
• Our app will also have a reporting feature so that the public can report any crime they have witnessed to the police and a cctv camera can be setup at that particular location.

---

References

Ş. Aktı, G.A. Tataroğlu, H.K. Ekenel, “*Vision-based Fight Detection from Surveillance Cameras*”, IEEE/EURASIP 9th International Conference on Image Processing Theory, Tools and Applications, Istanbul, Turkey, November 2019.
