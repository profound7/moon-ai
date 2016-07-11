package;

import moon.test.TestRunner;
import test.ai.neural.NeuralTest;
import test.ai.neural.OptimizerTest;


/**
 * ...
 * @author Munir Hussin
 */
class Test
{
    public static function main():Void
    {
        run();
        
        //var ntest = new NeuralTest();
        //ntest.exampleNeuronActivate();
        //ntest.exampleNeuronActivate();
    }
    
    public static function run():Void
    {
        var r:TestRunner = new TestRunner();
        
        // util
        r.add(new NeuralTest());
        r.add(new OptimizerTest());
        
        r.run();
    }
}

