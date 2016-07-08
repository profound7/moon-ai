package moon.ai.neural.networks;

import moon.ai.neural.Layer;
import moon.ai.neural.Network;
import moon.ai.neural.Trainer;

/**
 * ...
 * @author Munir Hussin
 */
class PerceptronNetwork extends Network
{
    public function new(layers:Array<Int>)
    {
        super();
        
        if (layers.length < 3)
            throw "Error: not enough layers (minimum 3) !!";
            
        var inputs:Int = layers.shift();  // first argument
        var outputs:Int = layers.pop();   // last argument
        
        // layers now contain hidden layers
        
        var input:Layer = new Layer(inputs);
        var hidden:Array<Layer> = [];
        var output:Layer = new Layer(outputs);
        
        var previous:Layer = input;
        
        // generate hidden layers
        for (size in layers)
        {
            var layer:Layer = new Layer(size);
            hidden.push(layer);
            previous.project(layer);
            previous = layer;
        }
        
        previous.project(output);
        
        // set layers of the neural network
        set({ input: input, hidden: hidden, output: output });
    }
    
}
