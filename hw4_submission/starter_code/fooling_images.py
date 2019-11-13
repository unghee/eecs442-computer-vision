import pickle
import matplotlib.pyplot as plt
from softmax import *
from common import *

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict

def gradient_ascent(model, target_class, init, learning_rate=1e-1):
    """
    Inputs:
    - model: Image classifier.
    - target_class: Integer, representing the target class the fooling image
      to be classified as.
    - init: Array, shape (1, Din), initial value of the fooling image.
    - learning_rate: A scalar for initial learning rate.
    
    Outputs:
    - image: Array, shape (1, Din), fooling images classified as target_class
      by model
    """
    
    image = init.copy()
    y = np.array([target_class])
    ###########################################################################
    # TODO: perform gradient ascent on your input image until your model      #
    # classifies it as the target class, get the gradient of loss with        #
    # respect to your input image by model.forwards_backwards(imgae, y, True) #
    ###########################################################################

    for i in range(500):
        dx,scores=model.forwards_backwards(image, y, True)
        image = image - dx*learning_rate
        if i%100 ==0:
            print(np.argmax(scores))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################   
    
    return image

def img_reshape(flat_img):
    # Use this function to reshape a CIFAR 10 image into the shape 32x32x3, 
    # this should be done when you want to show and save your image.
    return np.moveaxis(flat_img.reshape(3,32,32),0,-1)
    
    
def main():
    # Initialize your own model
    model = SoftmaxClassifier(hidden_dim = 200)
    config = {}
    target_class = None
    correct_image = None

    model.load('/Users/ungheelee/codes/eecs442-computer-vision/hw4_submission/starter_code/model_hidden')
    test_batch = unpickle("cifar-10-batches-py/test_batch")
    X_test = test_batch['data']
    Y_test = test_batch['labels']
    X_test_one = X_test[3]
    correct_image = X_test_one[None,:]

    norm_correct_image = (correct_image-np.mean(correct_image))/np.std(correct_image)

    target_class = 5 # 0
    ###########################################################################
    # TODO: load your trained model, correctly classified image and set your  #
    # hyperparameters, choose a different label as your target class          #
    ###########################################################################    
    fooling_image = gradient_ascent(model, target_class, init=norm_correct_image)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    difference_image = abs(fooling_image-norm_correct_image)*100.0
    fooling_image= fooling_image*np.std(correct_image)+np.mean(correct_image)
    correct_image = (norm_correct_image * np.std(correct_image)) + np.mean(correct_image)


    ###########################################################################
    # TODO: compute the (magnified) difference of your original image and the #
    # fooling image, save all three images for your report                    #
    ###########################################################################
    save_as_image( 'original' + '.jpg', correct_image)
    save_as_image( 'fooling' + '.jpg', fooling_image)
    save_as_image( 'difference' + '.jpg', difference_image)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


if __name__ == "__main__":
    main()
