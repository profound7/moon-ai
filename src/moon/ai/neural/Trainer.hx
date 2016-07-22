package moon.ai.neural;

import haxe.extern.EitherType;
import haxe.Timer;

/**
 * TODO: update to latest
 * https://github.com/cazala/synaptic/blob/master/src/trainer.js
 * 
 * @author Munir Hussin
 */
class Trainer
{
    public var network:Network;
    public var options:TrainerOptions;
    
    /*public var rate:EitherType<Float, Array<Float>>;
    public var iterations:Int;
    public var error:Float;
    public var shuffle:Bool;
    public var log:Int;
    public var cost:TrainerCostFunction;*/
    
    
    public function new(network:Network, ?options:TrainerOptions) 
    {
        this.network = network;
        resetOptions();
        setOptions(options);
    }
    
    public function resetOptions():Void
    {
        this.options =
        {
            rate: 0.2,
            iterations: 100000,
            error: 0.005,
            shuffle: false,
            log: 0,
            cost: TrainerCost.crossEntropy,
            customLog: null,
        };
    }
    
    public function setOptions(?opt:TrainerOptions):Void
    {
        if (opt != null)
        {
            if (Reflect.hasField(opt, "rate"))
                options.rate = opt.rate;
                
            if (Reflect.hasField(opt, "iterations"))
                options.iterations = opt.iterations;
                
            if (Reflect.hasField(opt, "error"))
                options.error = opt.error;
                
            if (Reflect.hasField(opt, "shuffle"))
                options.shuffle = opt.shuffle;
                
            if (Reflect.hasField(opt, "log"))
                options.log = opt.log;
                
            if (Reflect.hasField(opt, "cost"))
                options.cost = opt.cost;
                
            if (Reflect.hasField(opt, "customLog"))
                options.customLog = opt.customLog;
        }
    }
    
    // time in seconds, with fractions
    private static inline function getTime():Float
    {
        #if (flash || neko || php || js || cpp || sys)
            return Timer.stamp();
        #else
            return Date.now().getTime() / 1000;
        #end
    }
    
    // unbiased in-place shuffle algo, Fisher-Yates (Knuth) Shuffle
    public static function shuffle<T>(a:Array<T>):Void
    {
        var i:Int = a.length;
        var j:Int;
        var tmp:T;
        
        while (i-->1)
        {
            j = Math.floor(Math.random() * (i+1));
            
            tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    
    public static inline function log(message:String):Void
    {
        trace(message);
    }
    
    public function train(set:TrainingSet, ?opt:TrainerOptions):TrainingResults
    {
        var error:Float = 1;
        var iterations:Int = 0;
        var bucketSize:Int = 0;
        var input:Array<Float>;
        var output:Array<Float>;
        var target:Array<Float>;
        var currentRate:Float = 0;
        var rates:Array<Float> = null;
        
        setOptions(opt);
        
        
        if (Std.is(options.rate, Array))
        {
            rates = cast options.rate;
            bucketSize = Math.floor(options.iterations / rates.length);
        }
        else
        {
            currentRate = cast options.rate;
        }
        
        
        var start:Float = getTime();
        
        while (iterations < options.iterations && error > options.error)
        {
            error = 0;
            
            if (bucketSize > 0)
            {
                var currentBucket:Int = Math.floor(iterations / bucketSize);
                currentRate = rates[currentBucket];
            }
            
            for (data in set)
            {
                input = data.input;
                target = data.output;
                
                output = network.activate(input);
                network.propagate(currentRate, target);
                
                error += options.cost(target, output);
            }
            
            // check error
            iterations++;
            error /= set.length;
            
            if (options.customLog != null && iterations % options.customLog.every == 0)
            {
                options.customLog.log({
                    error: error,
                    iterations: iterations,
                    rate: currentRate,
                });
            }
            else if (options.log > 0 && iterations % options.log == 0)
            {
                log('iterations: $iterations error: $error rate: $currentRate');
            };
            
            if (options.shuffle)
                shuffle(set);
        }
        
        return
        {
            error: error,
            iterations: iterations,
            time: getTime() - start,
        }
    }
    
    // trains any given set to a network using a WebWorker
    public function workerTrain(set:Dynamic, callback:Dynamic, options:Dynamic):Void
    {
        throw "Not implemented!";
    }
    
    // trains an XOR to the network
    public function XOR(?options:TrainerOptions):TrainingResults
    {
        if (network.inputs() != 2 || network.outputs() != 1)
            throw "Error: Incompatible network (requires 2 inputs, 1 output)";
            
        var defaults:TrainerOptions =
        {
            iterations: 100000,
            log: 0,
            shuffle: true,
            cost: TrainerCost.meanSquaredError,
        }
        
        setOptions(defaults);
        setOptions(options);

        return train(
        [
            { input: [0, 0], output: [0] },
            { input: [1, 0], output: [1] },
            { input: [0, 1], output: [1] },
            { input: [1, 1], output: [0] }
        ]);
    }
    
    // trains the network to pass a Distracted Sequence Recall test
    public function DSR(?options:TrainerOptions):TrainingResults
    {
        throw "not yet ported";
    }
    
    // train the network to learn an Embeded Reber Grammar
    public function ERG(?options:TrainerOptions):TrainingResults
    {
        throw "not yet ported";
    }
}


typedef TrainerOptions =
{
    @:optional var rate:EitherType<Float, Array<Float>>;
    @:optional var iterations:Int;
    @:optional var error:Float;
    @:optional var shuffle:Bool;
    @:optional var log:Int;
    @:optional var cost:Array<Float>->Array<Float>->Float;
    @:optional var customLog:TrainerCustomLog;
}

typedef TrainerCustomLog =
{
    var every:Int;
    var log:TrainingLog->Void;
}

typedef TrainingResults =
{
    var error:Float;
    var iterations:Int;
    var time:Float;
}

typedef TrainingLog =
{
    var error:Float;
    var iterations:Int;
    var rate:Float;
}

typedef TrainingData =
{
    var input:Array<Float>;
    var output:Array<Float>;
}

typedef TrainingSet = Array<TrainingData>;

//typedef TrainerCostFunction = Array<Float>->Array<Float>->Float;



class TrainerCost
{
    public static function crossEntropy(target:Array<Float>, output:Array<Float>):Float
    {
        var crossentropy:Float = 0.0;
        for (i in 0...output.length)
            crossentropy -= (target[i] * Math.log(output[i] + 1e-15)) +
                ((1 - target[i]) * Math.log((1 + 1e-15) - output[i]));
                // +1e-15 is a tiny push away to avoid Math.log(0)
        return crossentropy;
    }
    
    public static function meanSquaredError(target:Array<Float>, output:Array<Float>):Float
    {
        var mse:Float = 0.0;
        for (i in 0...output.length)
            mse += Math.pow(target[i] - output[i], 2);
        return mse / output.length;
    }
    
    public static function binary(target:Array<Float>, output:Array<Float>):Float
    {
        var misses:Float = 0;
        for (i in 0...output.length)
            misses += Math.round(target[i] * 2) != Math.round(output[i] * 2) ? 1 : 0;
        return misses;
    }
}


