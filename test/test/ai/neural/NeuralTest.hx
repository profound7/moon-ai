package test.ai.neural;

import haxe.Json;
import moon.ai.neural.Layer;
import moon.ai.neural.Network;
import moon.ai.neural.networks.HopfieldNetwork;
import moon.ai.neural.networks.LiquidNetwork;
import moon.ai.neural.networks.LstmNetwork;
import moon.ai.neural.networks.PerceptronNetwork;
import moon.ai.neural.Neuron;
import moon.ai.neural.Trainer;
import moon.test.TestRunner;
import moon.test.TestCase;


/**
 * ...
 * @author Munir Hussin
 */
class NeuralTest extends TestCase
{
    private static inline var EPSILON:Float = 0.2;
    private static inline var EPSILON2:Float = 0.1;
    
    public static function main():Void
    {
        var r:TestRunner = new TestRunner();
        r.add(new NeuralTest());
        r.run();
    }
    
    
    public function exampleNeuronActivate():Void
    {
        var a:Neuron = new Neuron();
        var b:Neuron = new Neuron();
        a.project(b);
        
        a.activate(0.5);
        trace(b.activate());
        
        //assert.areNear(b.activate(), 0.5, EPSILON);
    }
    
    public function testNeuronPropagate():Void
    {
        var a:Neuron = new Neuron();
        var b:Neuron = new Neuron();
        a.project(b);

        var learningRate:Float = 0.3;

        for (i in 0...20000)
        {
            // when A activates 1
            a.activate(1);
            
            // train B to activate 0
            b.activate();
            b.propagate(learningRate, 0); 
        }
        
        // test it
        a.activate(1);
        //trace(b.activate());
        
        assert.isNear(b.activate(), 0.0, EPSILON);
    }
    
    public function exampleLayersGate():Void
    {
        var a:Layer = new Layer(5);
        var b:Layer = new Layer(4);
        
        var conn:LayerConnection = a.project(b);
        
        var c:Layer = new Layer(4);
        c.gate(conn, GateType.Input);
    }
    
    public function exampleLayersActivate():Void
    {
        var a:Layer = new Layer(5);
        var b:Layer = new Layer(3);
        
        a.project(b);
        
        a.activate([1, 0, 1, 0, 1]);
        trace(b.activate());
    }
    
    public function testLayersPropagate():Void
    {
        var a:Layer = new Layer(5);
        var b:Layer = new Layer(2);
        a.project(b);
        
        var learningRate:Float = 0.3;
        
        for (i in 0...20000)
        {
            // when A activates [1, 0, 1, 0, 1]
            a.activate([1, 0, 1, 0, 1]);
            
            // train B to activate [0,0]
            b.activate();
            b.propagate(learningRate, [1, 0]);
        }
        
        // test it
        a.activate([1, 0, 1, 0, 1]);
        //trace(b.activate());
        
        assert.areNear(b.activate(), [1.0, 0.0], EPSILON);
    }
    
    public function exampleNetworkActivate():Void
    {
        var inputLayer = new Layer(4);
        var hiddenLayer = new Layer(6);
        var outputLayer = new Layer(2);
        
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        
        var net = new Network({
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer
        });
        
        trace(net.activate([1, 0, 1, 0]));
    }
    
    public function testNetworkPropagate():Void
    {
        // create the network
        var inputLayer = new Layer(2);
        var hiddenLayer = new Layer(3);
        var outputLayer = new Layer(1);
        
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        
        var myNetwork = new Network({
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer,
        });
        
        // train the network
        var learningRate = 0.2; // 0.3 sometimes gives wrong result?
        for (i in 0...20000)
        {
            // 0,0 => 0
            myNetwork.activate([0, 0]);
            myNetwork.propagate(learningRate, [0]);
            
            // 0,1 => 1
            myNetwork.activate([0, 1]);
            myNetwork.propagate(learningRate, [1]);
            
            // 1,0 => 1
            myNetwork.activate([1, 0]);
            myNetwork.propagate(learningRate, [1]);
            
            // 1,1 => 0
            myNetwork.activate([1, 1]);
            myNetwork.propagate(learningRate, [0]);
        }
        
        // test cloning
        myNetwork = myNetwork.clone();
        
        // test the network
        var actual:Array<Float> =
        [
            myNetwork.activate([0, 0])[0],
            myNetwork.activate([0, 1])[0],
            myNetwork.activate([1, 0])[0],
            myNetwork.activate([1, 1])[0],
        ];
        
        //trace(myNetwork.activate([0, 0])); // [0.015020775950893527]
        //trace(myNetwork.activate([0, 1])); // [0.9815816381088985]
        //trace(myNetwork.activate([1, 0])); // [0.9871822457132193]
        //trace(myNetwork.activate([1, 1])); // [0.012950087641929467]
        
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    public function testTrainer1():Void
    {
        // create the network
        var inputLayer = new Layer(2);
        var hiddenLayer = new Layer(3);
        var outputLayer = new Layer(1);
        
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        
        var myNetwork = new Network({
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer,
        });
        
        var trainingSet:TrainingSet =
        [
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] },
        ];
        
        var trainer:Trainer = new Trainer(myNetwork);
        var results:TrainingResults = trainer.train(trainingSet);
        
        //trace("Training done!");
        //trace(results);
        
        // test the network
        var actual:Array<Float> =
        [
            myNetwork.activate([0, 0])[0],
            myNetwork.activate([0, 1])[0],
            myNetwork.activate([1, 0])[0],
            myNetwork.activate([1, 1])[0],
        ];
        
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    public function testTrainer2():Void
    {
        // create the network
        var inputLayer = new Layer(2);
        var hiddenLayer = new Layer(3);
        var outputLayer = new Layer(1);
        
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        
        var myNetwork = new Network({
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer,
        });
        
        
        var trainer:Trainer = new Trainer(myNetwork);
        var results:TrainingResults = trainer.XOR();
        
        //trace("Training done!");
        //trace(results);
        
        // test the network
        var actual:Array<Float> =
        [
            myNetwork.activate([0, 0])[0],
            myNetwork.activate([0, 1])[0],
            myNetwork.activate([1, 0])[0],
            myNetwork.activate([1, 1])[0],
        ];
        
        // using trainer.XOR(), since the cost is meanSquaredError, the epsilon is larger
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    public function testPperceptron():Void
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
        
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    public function testLtsm():Void
    {
        var nn:LstmNetwork = new LstmNetwork([2, 6, 1]);
        nn.trainer.XOR();
        
        // test the network
        var actual:Array<Float> =
        [
            nn.activate([0, 0])[0],
            nn.activate([0, 1])[0],
            nn.activate([1, 0])[0],
            nn.activate([1, 1])[0],
        ];
        
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    // this test sometimes fail
    public function testLiquid():Void
    {
        var nn:LiquidNetwork = new LiquidNetwork(2, 3, 1, 10, 3);
        nn.trainer.XOR();
        
        // test the network
        var actual:Array<Float> =
        [
            nn.activate([0, 0])[0],
            nn.activate([0, 1])[0],
            nn.activate([1, 0])[0],
            nn.activate([1, 1])[0],
        ];
        
        assert.areNear(actual, [0.0, 1.0, 1.0, 0.0], EPSILON);
    }
    
    // this test sometimes fail
    public function testHopfield():Void
    {
        // create a network for 10-bit patterns
        var nn:HopfieldNetwork = new HopfieldNetwork(10);
        
        var pattern1:Array<Float> = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        var pattern2:Array<Float> = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        
        
        // teach the network two different patterns
        nn.learn([
            pattern1,
            pattern2
        ]);
        
        // feed new patterns to the network and it will return the
        // most similar to the ones it was trained to remember
        var actual1:Array<Float> = nn.feed([0, 1, 0, 1, 0, 1, 0, 1, 1, 1]);
        var actual2:Array<Float> = nn.feed([1, 1, 1, 1, 1, 0, 0, 1, 0, 0]);
        
        //trace(actual1);     // [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        //trace(actual2);     // [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        
        assert.areNear(actual1, pattern1, EPSILON);
        assert.areNear(actual2, pattern2, EPSILON);
    }
}




