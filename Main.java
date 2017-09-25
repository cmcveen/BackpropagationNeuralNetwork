import java.io.*;
import java.util.*;

// Connor McVeen 

// Assignment 2 Main class
public class Main {
	
	

	public static void main(String[] args) throws IOException {
		double trainData [][] = readData("training");
		double testData [][] = readData("testing");
		
		double[][] inputTrainData = new double [trainData.length][trainData[0].length-1];
		double[][]outputTrainData = new double[trainData.length][10];
		double[][] inputTestData = new double[testData.length][testData[0].length-1];
		double[][] outputTestData = new double[testData.length][10];
		
		//Separates the inputs from the actual output into it's own 2D array
		for(int i =0;i<trainData.length;i++){
			for(int j = 0; j<trainData[i].length-1;j++){
				inputTrainData[i][j] = trainData[i][j];
			}
			
		}
		
		//Takes the outputs and separates it into a vector.
		for(int i =0;i<trainData.length;i++){
			for(int j = 0; j<10;j++){
				if(trainData[i][trainData[i].length-1]==j){
					outputTrainData[i][j] = 1.0;
				}else{
					outputTrainData[i][j] = 0.0;
				}
			}			
					
		}
		
		//Separates the inputs from the actual output into it's own 2D array
		for(int i =0;i<testData.length;i++){
			for(int j = 0; j<testData[i].length-1;j++){
				inputTestData[i][j] = testData[i][j];
			}
			
		}		
		//Takes the outputs and separates it into a vector.
		for(int i =0;i<testData.length;i++){
			for(int j = 0; j<10;j++){
				if(testData[i][testData[i].length-1]==j){
					outputTestData[i][j]=1.0;
				}else{
					outputTestData[i][j]= 0.0;
				}
			}	
		}
		
		//Defines how many neurons per layer
		int [] neurons = {64,100,10};
		
		System.out.println("====WITHOUT MOMENTUM===");
		
		Network neuralNet = new Network(neurons, inputTrainData, outputTrainData, 0);
		
		neuralNet.run();
		
		neuralNet.testPhase(inputTestData, outputTestData);
		
		System.out.println("====WITH MOMENTUM===");
		
		neuralNet = new Network(neurons, inputTrainData,outputTrainData,0.2);
		
		neuralNet.run();
		
		neuralNet.testPhase(inputTestData, outputTestData);
	
		
	}

	//reads the data and returns it as a 2 dimensional double array
	private static double[][] readData(String filename) throws IOException {
		
		BufferedReader dataBR = new BufferedReader( new FileReader(filename+".txt"));

		String line ="";
		
		ArrayList<double[]> dataArr = new ArrayList<double[]>();
		
		
		while ((line = dataBR.readLine()) != null){
			double [] data = new double[65];
			
			for(int i =0; i<65;i++){
				String [] value = line.split(",",65);
				data[i] = Double.valueOf(value[i]);
			}
			
			dataArr.add(data);
		}
		
		dataBR.close();
		
		double [][] finalData = new double[dataArr.size()][];
		finalData = dataArr.toArray(finalData);
		
		
		return finalData;
	}

}
