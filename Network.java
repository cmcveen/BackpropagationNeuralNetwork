import java.io.*;


//The network class that brings everything together, and runs through a neural net utilizing a backpropagation algorithm. 
public class Network {

	//Error function variable
	private double globalError;

	//minimum error function defined by user?
	private double minError;

	//expected output pattern
	private double[][] expectedOutput;
	
	private double[][] testingExpectedOutput;

	//input pattern for set of samples
	private double[][] inputs;
	
	private double[][] testingInputs;

	//learning rate
	private double learningRate;

	//momentum
	private double momentum;

	//number of layers in network (input, hidden, output)
	private int numOfLayers;

	//number of training sets?
	private int numOfSamples;
	
	private int testingNumOfSamples;

	//current training set/sample that is used?
	private int sampleNum;

	//Maximum num of epochs before training stops
	private long maxEpochs;

	public Layer layer[];

	public double actualOutput[][];
	public double testingActualOutput [][];


	//node activations calculation
	public void feedForward(){


		//Since no weights contribute to the output from the input, assign the input
		//vector from the input layer to all the node in the first hidden layer
		for(int i=0;i<layer[0].node.length;i++){
			layer[0].node[i].output=layer[0].inputs[i];
		}

		layer[1].inputs=layer[0].inputs;

		for(int i = 1; i<numOfLayers;i++){
			layer[i].feedForward();

			//unless we have reached the last layer, assign the layer i's output vector
			// to the (i+1) layer's input vector
			if(i != numOfLayers-1){
				layer[i+1].inputs=layer[i].outputVector();
			}
		}
	}

	//Backpropagated the network output error through the network to update the weight values
	public void updateWeights(){
		calculatenodeErrors();
		backPropagateError();
	}

	private void calculatenodeErrors(){
		int outputLayer;
		double sum;

		outputLayer = numOfLayers-1;

		//calc all output signal error
		for(int i=0;i<layer[outputLayer].node.length;i++){
			layer[outputLayer].node[i].nodeError = (expectedOutput[sampleNum][i] - layer[outputLayer].node[i].output)*
					layer[outputLayer].node[i].output*(1-layer[outputLayer].node[i].output);
		}

		// calculate signal error for all nodes in the hidden layer (back propagate errors)
		for (int i = numOfLayers-2; i>0;i--){
			for(int j = 0;j<layer[i].node.length;j++){
				sum = 0;
				for(int k =0;k<layer[i+1].node.length;k++){
					sum = sum +layer[i+1].node[k].weights[j]*layer[i+1].node[k].nodeError;
				}

				layer[i].node[j].nodeError= layer[i].node[j].output*(1-layer[i].node[j].output)*sum;
			}
		}
	}


	private void backPropagateError(){
		//update weights

		for(int i = numOfLayers-1;i>0;i--){
			for(int j=0;j<layer[i].node.length;j++){
				//calculate bias weight difference to node j
				layer[i].node[j].thresholdDifference = learningRate*layer[i].node[j].nodeError+momentum*layer[i].node[j].thresholdDifference;

				//update bias weight to node j
				layer[i].node[j].threshold=layer[i].node[j].threshold+layer[i].node[j].thresholdDifference;

				//update weights
				for(int k =0;k<layer[i].inputs.length;k++){
					//calculate weight difference between node j and k
					layer[i].node[j].weightDifference[k] = learningRate*layer[i].node[j].nodeError*layer[i-1].node[k].output+momentum*layer[i].node[j].weightDifference[k];

					//update weight between node j and k
					layer[i].node[j].weights[k]=layer[i].node[j].weights[k]+layer[i].node[j].weightDifference[k]; 
				}
			}
		}
	}


	private void calculateGlobalError(){
		globalError =0;

		for(int i =0;i<numOfSamples;i++){
			for(int j=0;j<layer[numOfLayers-1].node.length;j++){
				globalError = globalError + 0.5*(Math.pow(expectedOutput[i][j]-actualOutput[i][j], 2));
			}
		}
	}


	public Network(int[] numOfNodes, double[][] inputSamples, double[][] outputSamples, double momentum){
		//assign variables
		numOfSamples = inputSamples.length;
		this.minError = 0;
		this.learningRate = 0.1;
		this.momentum = momentum;
		numOfLayers=numOfNodes.length;
		this.maxEpochs= 150;

		//create layers
		layer = new Layer[numOfLayers];

		//assign the number of node to the input layer
		layer[0] = new Layer(numOfNodes[0],numOfNodes[0]);

		//assign number of nodes to each layer
		for(int i =1; i<numOfLayers;i++){
			layer[i]=new Layer(numOfNodes[i],numOfNodes[i-1]);
		}

		inputs = new double[numOfSamples][layer[0].node.length];
		expectedOutput = new double[numOfSamples][layer[numOfLayers-1].node.length];
		actualOutput = new double[numOfSamples][layer[numOfLayers-1].node.length];

		//Assign input set
		for(int i = 0;i<numOfSamples;i++){
			for(int j= 0; j<layer[0].node.length;j++){
				inputs[i][j] = inputSamples[i][j];
			}
		}

		//Assign the output set
		for(int i = 0; i<numOfSamples;i++){
			for(int j=0;j<layer[numOfLayers-1].node.length;j++){
				expectedOutput[i][j] = outputSamples[i][j];
			}
		}

	}


	public void train(){
		long k =0;
		do{
			//for each data sample
			for(sampleNum =0; sampleNum < numOfSamples; sampleNum++){
				for(int i = 0; i< layer[0].node.length;i++){
					layer[0].inputs[i] = inputs[sampleNum][i];
				}
				feedForward();
				//assign calculated output vector from network to actual output
				for(int i =0;i<layer[numOfLayers-1].node.length;i++){
					actualOutput[sampleNum][i] = layer[numOfLayers-1].node[i].output;
				}
				updateWeights();			
			}
			k++;
			calculateGlobalError();
		}while((globalError>minError)&&(k<maxEpochs));
		System.out.println("It took " + k + " epochs, and the final global error is "+ globalError);
	}
	public void testPhase(double[][] data, double[][]outputs) throws IOException{
		
		//Assign the testing inputs and outputs to new arrays
		testingNumOfSamples = data.length;
		testingInputs = new double [testingNumOfSamples][layer[0].node.length];
		testingExpectedOutput = new double [testingNumOfSamples][layer[numOfLayers-1].node.length];
		testingActualOutput = new double [testingNumOfSamples][layer[numOfLayers-1].node.length];
		
		
		double count =0;
		double errors =0;
		int sums;
		double [][] result = new double [outputs.length][outputs[0].length];
		//Assignments
		for(int i = 0;i<testingNumOfSamples;i++){
			for(int j=0;j<layer[0].node.length;j++){
				testingInputs[i][j] = data[i][j];
			}
		}
		//Assignments
		for(int i = 0;i<testingNumOfSamples;i++){
			for (int j = 0; j < layer[numOfLayers-1].node.length; j++){
				testingExpectedOutput[i][j] = outputs[i][j];
			}
		}
		
		//for each data sample
		for(int sampleNum =0; sampleNum<testingNumOfSamples;sampleNum++){
			
			//Assign the input data to the input nodes, and feed the data through the network before taking the outputs from
			//the output nodes, and placing that in an array
			for(int i =0; i<layer[0].node.length;i++){
				layer[0].inputs[i] = testingInputs[sampleNum][i];
			}
			feedForward();
			
			for(int i =0;i<layer[numOfLayers-1].node.length;i++){
				testingActualOutput[sampleNum][i] = Math.round(layer[numOfLayers-1].node[i].output);
			}
			
		}
		
		//For each output, subtracted the calculated outputs from what the outputs should be and store the result in a new array
		for(int i =0 ;i<testingExpectedOutput.length;i++){
			for(int j =0; j<testingExpectedOutput[i].length;j++){
				result[i][j] = testingExpectedOutput[i][j] - testingActualOutput[i][j];							
			}		
		}
		//sum the results for each data set, if the sum is zero than the data has been correctly identified
		//do this for each sample
		for(int i = 0; i<result.length;i++){
			count++;
			sums =0;
			
			for(int j=0;j<result[i].length;j++){
				sums+= result[i][j];
			}
			if(sums!=0){
				errors++;
			}
		}
		
		System.out.println("Total errors: " + errors+"/"+count);
		System.out.println("Percent correct:"+ ((1-(errors/count))*100));
		
		
		writeResults(testingInputs, testingExpectedOutput, testingActualOutput, errors, count);
	}
	
	public void run(){
		train();
	}
	
	private static void writeResults(double[][]data, double[][]outputReal, double[][] outputCalcd, double errors, double count)throws IOException{
		int [] realOutput = new int[outputReal.length];
		int[] calcdOutput = new int[outputCalcd.length];
		
		//Organize the output data back into an integer value
		for(int i =0;i<outputReal.length;i++){
			for(int j=0;j<outputReal[i].length;j++){
				if(outputReal[i][j]==1.0){
					realOutput[i]=j;
				}
			}
		}
		
		for(int i =0;i<outputCalcd.length;i++){
			for(int j=0;j<outputCalcd[i].length;j++){
				if(outputCalcd[i][j]==1.0){
					calcdOutput[i]=j;
				}
			}
		}
		
		//Set the filename
		String filename ="predictedResults.txt";
		
		//If the file already exists, append to the file instead of overwriting it
		//Writes the data in the exact same format as was given, and has the classification
		//that was calculated by the network added on the end after a hyphen
		BufferedWriter outputWriter = new BufferedWriter (new FileWriter(filename, true));
		outputWriter.write("============NEW DATA ENTRY============");
		outputWriter.newLine();
		for(int i =0;i<data.length;i++){
			for(int j=0;j<data[i].length;j++){
				outputWriter.write((int)data[i][j]+",");
			}
			outputWriter.write(realOutput[i]+" - "+ calcdOutput[i]);
			outputWriter.newLine();
		}
		outputWriter.newLine();
		outputWriter.write("Total errors: " + (int)errors+"/"+(int)count);
		outputWriter.write(" Percent correct:"+ ((1-(errors/count))*100));
		outputWriter.flush();
		outputWriter.close();
		
	}
	
}
















