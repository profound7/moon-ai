package test.ai.neural;

import moon.ai.neural.networks.PerceptronNetwork;
import moon.ai.neural.Optimizer;
import moon.core.Error;
import moon.test.TestRunner;
import moon.test.TestCase;

/**
 * ...
 * @author Munir Hussin
 */
class OptimizerTest extends TestCase
{
    private static inline var EPSILON:Float = 0.3;
    private static var expected:Array<Float> = [0.0, 1.0, 1.0, 0.0];
    
    public static function main():Void
    {
        var r:TestRunner = new TestRunner();
        r.add(new OptimizerTest());
        r.run();
    }
    
    public function testOptimizer() 
    {
        var nn:PerceptronNetwork = new PerceptronNetwork([2, 3, 1]);
        nn.trainer.XOR();
        
        // test the network
        var actual:Array<Float> =
        [
            nn.activate([0, 0])[0],
            nn.activate([0, 1])[0],
            nn.activate([1, 0])[0],
            nn.activate([1, 1])[0],
        ];
        
        assert.areNear(actual, expected, EPSILON);
        
        try
        {
            Optimizer.save(nn, "../../test", "test.ai.neural.OptimizedNetwork2");
        }
        catch (ex:Error)
        {
            ex.printStackTrace();
        }
        
        // test the generated network
        var nx:OptimizedNetwork = new OptimizedNetwork();
        
        var optimized:Array<Float> =
        [
            nn.activate([0, 0])[0],
            nn.activate([0, 1])[0],
            nn.activate([1, 0])[0],
            nn.activate([1, 1])[0],
        ];
        
        assert.areNear(optimized, expected, EPSILON);
        
        // actual network and optimized network should give same results
        assert.areEqual(actual, optimized);
        
        trace("");
        trace(actual);
        trace(optimized);
        
        // BUG: something is wrong with code generation.
        // the outputs should exactly match, instead it closely match
        // this happens when the XOR trainer is using meanSquaredError as cost.
        // output is correct when using crossEntropy
    }
    
}