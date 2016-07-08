package moon.ai.neural.networks;

import moon.ai.neural.Layer;
import moon.ai.neural.Network;

/**
 * ...
 * @author Munir Hussin
 */
class LstmNetwork extends Network
{

    public function new(layers:Array<Int>) 
    {
        super();
        
        if (layers.length < 3)
            throw "Error: not enough layers (minimum 3) !!";
            
        var inputs:Int = layers.shift();
        var outputs:Int = layers.pop();
        
        var inputLayer:Layer = new Layer(inputs);
        var hiddenLayers:Array<Layer> = [];
        var outputLayer:Layer = new Layer(outputs);
        
        var previous:Layer = null;
        
        // generate layers
        for (size in layers)
        {
            // generate memory blocks (memory cell and respective gates)
            var inputGate:Layer = new Layer(size).set({ bias: 1 });
            var forgetGate:Layer = new Layer(size).set({ bias: 1 });
            var memoryCell:Layer = new Layer(size);
            var outputGate:Layer = new Layer(size).set({ bias: 1 });
            
            hiddenLayers.push(inputGate);
            hiddenLayers.push(forgetGate);
            hiddenLayers.push(memoryCell);
            hiddenLayers.push(outputGate);
            
            // connections from input layer
            var input:LayerConnection = inputLayer.project(memoryCell);
            inputLayer.project(inputGate);
            inputLayer.project(forgetGate);
            inputLayer.project(outputGate);
            
            var cell:LayerConnection = null;
            
            // connections from previous memory-block layer to this one
            if (previous != null)
            {
                cell = previous.project(memoryCell);
                previous.project(inputGate);
                previous.project(forgetGate);
                previous.project(outputGate);
            }
            
            // connections from memory cell
            var output:LayerConnection = memoryCell.project(outputLayer);
            
            // self-connection
            var self:LayerConnection = memoryCell.project(memoryCell);
            
            // peepholes
            memoryCell.project(inputGate, ConnectionType.OneToOne);
            memoryCell.project(forgetGate, ConnectionType.OneToOne);
            memoryCell.project(outputGate, ConnectionType.OneToOne);
            
            // gates
            inputGate.gate(input, GateType.Input);
            forgetGate.gate(self, GateType.OneToOne);
            outputGate.gate(output, GateType.Output);
            
            if (previous != null)
                inputGate.gate(cell, GateType.Input);
                
            previous = memoryCell;
        }
        
        // input to output direct connection
        inputLayer.project(outputLayer);
        
        // set the layers of the neural network
        set({ input: inputLayer, hidden: hiddenLayers, output: outputLayer });
    }
    
}
