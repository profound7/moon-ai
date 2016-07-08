package moon.ai.neural.networks;

import moon.ai.neural.Network;
import moon.ai.neural.Neuron.NeuronConnection;

/**
 * ...
 * @author Munir Hussin
 */
class LiquidNetwork extends Network
{
    
    public function new(inputs:Int, hidden:Int, outputs:Int, connections:Int, gates:Int)
    {
        super();
        
        // create layers
        var inputLayer = new Layer(inputs);
        var hiddenLayer = new Layer(hidden);
        var outputLayer = new Layer(outputs);
        
        // make connections and gates randomly among the neurons
        var neurons:Array<Neuron> = hiddenLayer.neurons();
        var connectionList:Array<NeuronConnection> = [];
        
        for (i in 0...connections)
        {
            // connect two random neurons
            var from:Int = Math.floor(Math.random() * neurons.length);
            var to:Int = Math.floor(Math.random() * neurons.length);
            var connection:NeuronConnection = neurons[from].project(neurons[to]);
            connectionList.push(connection);
        }
        
        for (j in 0...gates)
        {
            // pick a random gater neuron
            var gater:Int = Math.floor(Math.random() * neurons.length);
            // pick a random connection to gate
            var connection:Int = Math.floor(Math.random() * connectionList.length);
            // let the gater gate the connection
            neurons[gater].gate(connectionList[connection]);
        }
        
        // connect the layers
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        
        // set the layers of the network
        set({ input: inputLayer, hidden: [hiddenLayer], output: outputLayer });
    }
    
}