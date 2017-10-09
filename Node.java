
import java.util.*;


// The node class which describes a node in the network
public class Node {

	//output from node
	public double output;
	
	//weights from other nodes to this node
	public double weights[];
	
	//threshold for node
	public double threshold;
	
	//weight difference between iterations
	
	public double weightDifference[];
	
	//threshold difference between iterations
	
	public double thresholdDifference;
	
	//output  error
	public double nodeError;
	
	
	//Assigns randomly generated number between -1 and 1 to threshold and weights to node
	
	private void intialiseWeights(){
		Random rand = new Random();
		
		
		threshold = 2*rand.nextDouble()-1;
		
		thresholdDifference =0;
		
		for(int i=0;i<weights.length;i++){
			weights[i] =2*rand.nextDouble()-1;
			
			//Difference intially assigned as zero
			weightDifference[i]=0;
		}
	}
	
	//Create array of weights with same size of inputs to the node
	//Also creates and array to hold weight differences, same size as inputs to the node
	//Finally, intializes the weights and thresholds
	public Node (int numOfNodes){
		weights = new double[numOfNodes];
		
		weightDifference = new double[numOfNodes];
		
		intialiseWeights();
	}
	
	

	
	
}
