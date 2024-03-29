## ReadMe file ##

Environment
  Operation system: OS
  Programming language: Python version 3.8.0
  Package installed:
	pandas==1.4.1
	numpy==1.23.2
	scikit-learn==1.1.0
	matplotlib==3.5.1

Instruction
1. Make sure that training and test dataset file is the same directory as main.py(implemented code)
2. In if __name__ == "__main__": 
	there are 2 main steps: training, testing. 
	then, in testing process there are 2 functions:  
	    - run_test(filename, x_train, y_train)
        	filename = file that you want to test
        	x_train, y_train = fixed
          - create_result_file(predict_y, accuracy_training, f1_training, filename)
        	filename = file name that you want to create. 

	after run this function, it will produce a student_id.csv file that contains test result and accuracy and F1 of training dataset

	
		
       