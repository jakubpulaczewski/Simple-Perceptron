import numpy as np

# Constants

TRAIN_FILE = "train.data"
TEST_FILE = "test.data"
CLASS_ONE = 'class-1'
CLASS_TWO = 'class-2'
CLASS_THREE = 'class-3'
# This consant is used for Question 7
GAMMA = 0.00


class Perceptron():
    ''''
    This class shall train a perceptron and then be evaluted on another new set of data
    and this will be given in terms of precision, recall, accuracy and score.

    @input_size: the number of weight of connections
    @n_iters: number of iterations that the perceptron will train on
    '''

    def __init__(self, input_size=4, n_iters=20):
        self.weights = np.zeros(input_size)
        self.n_iters = n_iters
        self.bias = 0

    '''
    This method calculates the dot product between the weights and input features + bias
  
    @x: the input values
    '''
    def calculate(self, x):
        return self.weights.T.dot(x) + self.bias

    '''
    This method predicts the label for a given sample. 
  
    @input: sum of values
    '''
    def predict(self, input):
        return 1.0 if input >= 0.0 else -1.0

    '''
    This method evaluates each sample with its predicted class and this is compared
    with the desired output to give some evaluation in terms of precision, recall, 
    accuracy and score.
  
    @input_array: the array that contains all datasets
    '''
    def test(self, input_array):
        # Initiliation the variables to 0
        true_positives = true_negatives = false_positives = false_negatives = precision = recall = accuracy = score = 0
        # The dataset is split into two arrays
        inputs = input_array[:, 0:4]
        desired_output = input_array[:, 4:5]
        # Evaluation using TP (True Positives), TN (True Negatives), FP (False Positives), FN (False Negatives)
        for i in range(desired_output.shape[0]):
            summation = self.calculate(inputs[i, :])
            predicted = self.predict(summation)
            if ((predicted == 1) & (desired_output[i] == 1)):
                true_positives += 1
            elif ((predicted == -1) & (desired_output[i] == -1)):
                true_negatives += 1
            elif ((predicted == 1) & (desired_output[i] == -1)):
                false_positives += 1
            elif ((predicted == -1) & (desired_output[i] == 1)):
                false_negatives += 1
        # Zero Divison Error Exception Handling (using if statements)
        precision = 0 if ((true_positives == 0) or ((true_positives + false_positives) == 0)) else (
                    true_positives / (true_positives + false_positives))
        recall = 0 if ((true_positives == 0) or ((true_positives + false_positives) == 0)) else true_positives / (
                    true_positives + false_negatives)
        accuracy = 0 if (((true_positives + true_negatives) == 0) or (
                    (true_positives + true_negatives + false_positives + false_negatives) == 0)) else (
                    true_positives + true_negatives) / ( true_positives + true_negatives + false_positives + false_negatives)
        score = 0 if (((precision * recall) == 0) or ((precision + recall) == 0)) else (2 * precision * recall) / (precision + recall)
        # Outputting the results
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Accuracy: " + str(accuracy))
        print("F-score " + str(score))

    '''
    This method trains the perceptron.
  
    @input_array: the array that contains all datasets
    @gamma: the l2 regularisation only for question/task 7
    '''
    def train(self, input_array, gamma=0):
        # The dataset is split into two arrays (inputs) and (desired outputs)
        inputs = input_array[:, 0:4]
        desired_output = input_array[:, 4:5]
        for _ in range(self.n_iters):
            for i in range(desired_output.shape[0]):
                # Calculates the dot product + bias
                output = self.calculate(inputs[i, :])
                if (desired_output[i] * output <= 0):
                    self.weights = self.weights + desired_output[i] * inputs[i, :] + gamma * (np.sum(self.weights) ** 2)
                    self.bias = self.bias + desired_output[i]


'''
This method reads the file.

@file: the name of file
@return: the dataset
'''
def readData(file=TRAIN_FILE):
    with open(file) as myfile:
        data = myfile.read()
        data = data.splitlines()
        data = [x.split(',') for x in data]
        data = np.array(data)
        return data


'''
This method categorizes the dataset in 3 different classes such as class1, class2 and class3.

@file: the name of file
@return: the array for a given class
'''
def categorizeClass(class_number, data):
    r1, c1 = np.where(data == class_number)
    array = np.array(data[r1])
    # It changes the last column to 0 so that it has dtype of float and thus can be multipled with self.weights for dot product in calculate method
    array[:, -1] = 0
    array = np.array(array, dtype=float)
    return array


'''
This method adjusts the desired output of both classes such that each class have 
different desired output

@class_A: first class
@class_B: second class
'''
def adjust_DesiredOutput(class_A, class_B):
    class_A[:, -1] = -1
    class_B[:, -1] = 1
    merged = np.concatenate([class_A, class_B])
    return merged


'''
This method just like adjust_DesiredOutput, adjusts desired output to classes
using 1 vs rest approach. In which, one class is give unique desired output and
the rest classes have the same desired output.

@classification_no: the number of class to be assigned unique desired output
'''
def multiClass_DesiredOutput(classification_no, class_A, class_B, class_C):
    if (classification_no == 1):
        class_A[:, -1] = 1
        class_B[:, -1] = -1
        class_C[:, -1] = -1
    elif (classification_no == 2):
        class_A[:, -1] = -1
        class_B[:, -1] = 1
        class_C[:, -1] = -1
    elif (classification_no == 3):
        class_A[:, -1] = -1
        class_B[:, -1] = -1
        class_C[:, -1] = 1
    merged = np.concatenate([class1, class2, class3])
    return merged


def question4():
    ''' -------------------- Class 1 and 2 --------------------
    It creates a new object Perceptron and assigns a desired output for both class 1 and class 2.
    Then it tests the classification with another dataset and outputs the evaluation.
    '''
    print("-------------------- Class 1 and 2 --------------------")
    # Training Class 1 and Class 2
    perceptron1 = Perceptron()
    merged = adjust_DesiredOutput(class1, class2)
    perceptron1.train(merged)
    # Testing Class 1 and Class 2
    merged_test = adjust_DesiredOutput(class1_test, class2_test)
    perceptron1.test(merged_test)

    ''' -------------------- Class 2 and 3 --------------------
    It creates a new object Perceptron and assigns a desired output for both class 2 and class 3.
    Then it tests the classification with another dataset and outputs the evaluation.
    '''
    print("-------------------- Class 2 and 3 --------------------")
    # Training Class 2 and Class 3
    perceptron2 = Perceptron()
    merged = adjust_DesiredOutput(class2, class3)
    perceptron2.train(merged)
    # Testing Class 1 and Class 2
    merged_test = adjust_DesiredOutput(class2_test, class3_test)
    perceptron2.test(merged_test)

    ''' -------------------- Class 1 and 3 --------------------
    It creates a new object Perceptron and assigns a desired output for both class 1 and class 3.
    Then it tests the classification with another dataset and outputs the evaluation.
    '''
    print("-------------------- Class 1 and 3 --------------------")
    # Training Class 2 and Class 3
    perceptron3 = Perceptron()
    merged = adjust_DesiredOutput(class1, class3)
    perceptron3.train(merged)
    # Testing Class 1 and Class 2
    merged_test = adjust_DesiredOutput(class1_test, class3_test)
    perceptron3.test(merged_test)


def question6and7():
    '''  -------------------- Class 1 vs the rest --------------------
    It creates a new object Perceptron and assigns a desired output for class 1 and for the rest it assigns
    the same desired output so that class 1 have unique desired output and then it outputs the evaluation of the classification.
    '''
    print("-------------------- Class 1 vs the rest --------------------")
    # Training the Perceptron
    rest1 = Perceptron()
    merged = multiClass_DesiredOutput(1, class1, class2, class3)
    rest1.train(merged, GAMMA)
    # Testing
    merged_test = multiClass_DesiredOutput(1, class1_test, class2_test, class3_test)
    rest1.test(merged_test)
    '''  -------------------- Class 2 vs the rest -------------------- '''
    print("-------------------- Class 2 vs the rest --------------------")
    # Training the Perceptron
    rest2 = Perceptron()
    merged = multiClass_DesiredOutput(2, class1, class2, class3)
    rest2.train(merged, GAMMA)
    # Testing
    merged_test = multiClass_DesiredOutput(2, class1_test, class2_test, class3_test)
    rest2.test(merged_test)
    '''  -------------------- Class 3 vs the rest -------------------- '''
    print("-------------------- Class3 vs the rest --------------------")
    # Training the Perceptron
    rest3 = Perceptron()
    merged = multiClass_DesiredOutput(3, class1, class2, class3)
    rest3.train(merged, GAMMA)

    merged_test = multiClass_DesiredOutput(3, class1_test, class2_test, class3_test)
    rest3.test(merged_test)


if __name__ == '__main__':
    # Categorization of data into 3 different classes (class1, class2 and class3) for training the perceptron
    class1 = categorizeClass(CLASS_ONE, readData())
    class2 = categorizeClass(CLASS_TWO, readData())
    class3 = categorizeClass(CLASS_THREE, readData())

    # Categorization of data into 3 different classes for testing the perceptron with new dataset
    class1_test = categorizeClass(CLASS_ONE, readData(TEST_FILE))
    class2_test = categorizeClass(CLASS_TWO, readData(TEST_FILE))
    class3_test = categorizeClass(CLASS_THREE, readData(TEST_FILE))
    # Question 4
    question4()
    # Question 6 and 7
    question6and7()
