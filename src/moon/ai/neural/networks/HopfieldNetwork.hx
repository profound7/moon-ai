package moon.ai.neural.networks;

import moon.ai.neural.Layer;
import moon.ai.neural.Network;
import moon.ai.neural.Trainer;

/**
 * ...
 * @author Munir Hussin
 */
class HopfieldNetwork extends Network
{
    
    public function new(size:Int) 
    {
        super();
        
        var inputLayer:Layer = new Layer(size);
        var outputLayer:Layer = new Layer(size);
        
        inputLayer.project(outputLayer, LayerConnectionType.AllToAll);
        
        set({ input: inputLayer, hidden: [], output: outputLayer });
    }
    
    public function learn(patterns:Array<Array<Float>>):TrainingResults
    {
        var set:TrainingSet = [];
        
        for (p in patterns)
            set.push({ input: p, output: p });
            
        return trainer.train(set,
        {
            iterations: 500000,
            error: 0.00005,
            rate: 1,
        });
    }
    
    public function feed(pattern:Array<Float>):Array<Float>
    {
        var output:Array<Float> = activate(pattern);
        var pattern:Array<Float> = [];
        
        for (i in 0...output.length)
            pattern[i] = output[i] > 0.5 ? 1 : 0;
        
        return pattern;
    }
    
}
