


// The layer class that describes a layer in the network
public class Layer {

	private double net;
	
	//inputs from previous layer to current layer
	public double inputs[];
	
	//nodes in current layer
	public Node node[];
	
	
	//feed forward function called so that outputs for all the nodes in the current
	//layer are calculated
	
	
	
	public void feedForward(){
		for(int i=0;i<node.length;i++){
			net = node[i].threshold;
			
			for (int j =0;j<node[i].weights.length;j++){
				net = net + inputs[j]* node[i].weights[j];
			}
			node[i].output =sigmoid(net);
		}
	}
	
	//sigmoid function calculates activation/output from current node
	private double sigmoid(double net){
		return 1/(1+Math.exp(-net));
	}
	
	//return the output from all node in layer
	
	public double[] outputVector(){
		double vector[];
		vector = new double [node.length];
		
		for(int i=0;i<node.length;i++){
			vector[i]=node[i].output;
		}
		return vector;
	}
	
	public Layer (int numOfNodes, int numOfInputs){
		node = new Node[numOfNodes];
		
		for (int i=0;i<numOfNodes;i++){
			node[i]= new Node(numOfInputs);
		}
		inputs = new double[numOfInputs];
	}
	
	public Node[] getNodes(){
		return node;
	}
}
