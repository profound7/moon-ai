package moon.ai.neural;

import moon.ai.neural.Layer.ConnectionType;
import moon.ai.neural.Layer.GateType;
import moon.ai.neural.Layer.ILayerProjectable;
import moon.ai.neural.Layer.LayerConnection;
import moon.ai.neural.Network.NetworkLayers;
import moon.ai.neural.Neuron.NeuronConnection;

/**
 * https://github.com/cazala/synaptic/wiki/Networks
 * @author Munir Hussin
 */
class Network implements ILayerProjectable implements INetwork
{
    public var layers:NetworkLayers;
    public var optimized:NetworkOptimized;
    public var optimizeEnabled:Bool;
    public var trainer:Trainer;
    
    
    public function new(?layers:NetworkLayers) 
    {
        this.layers = layers != null ? layers :
        {
            input: null,
            hidden: [],
            output: null,
        };
        
        this.optimizeEnabled = false;
        this.trainer = new Trainer(this);
    }
    
    
    
    // feed-forward activation of all the layers to produce an ouput
    public function activate(input:Array<Float>):Array<Float>
    {
        if (optimizeEnabled == false)
        {
            layers.input.activate(input);
            
            for (layer in layers.hidden)
                layer.activate();
                
            return layers.output.activate();
        } 
        else 
        {
            if (optimized == null)
                optimize();
                
            return optimized.activate(input);
        }
    }
    
    // back-propagate the error thru the network
    public function propagate(rate:Float=0.1, target:Array<Float>):Void
    {
        if (optimizeEnabled == false)
        {
            layers.output.propagate(rate, target);
            
            var reverse:Array<Layer> = layers.hidden.copy();
            reverse.reverse();
            
            for (layer in reverse)
                layer.propagate(rate);
        }
        else 
        {
            if (optimized == null)
                optimize();
                
            optimized.propagate(rate, target);
        }
    }
    
    // project a connection to another unit (either a network or a layer)
    public function project(unit:ILayerProjectable, ?type:ConnectionType, ?weight:Float):LayerConnection
    {
        if (optimized != null)
            optimized.reset();
            
        return layers.output.project(unit.getProjectableLayer(), type, weight);
    }
    
    // interface to get a projectable layer
    public function getProjectableLayer():Layer
    {
        return layers.input;
    }
    
    // let this network gate a connection
    public function gate(connection:LayerConnection, type:GateType):Void
    {
        if (optimized != null)
            optimized.reset();
            
        layers.output.gate(connection, type);
    }
    
    // clear all elegibility traces and extended elegibility traces
    // (the network forgets its context, but not what was trained)
    public function clear():Void
    {
        restore();
        
        var inputLayer:Layer = layers.input;
        var outputLayer:Layer = layers.output;
        
        inputLayer.clear();
        
        for (hiddenLayer in layers.hidden)
            hiddenLayer.clear();
            
        outputLayer.clear();
        
        if (optimized != null)
            optimized.reset();
    }
    
    // reset all weights and clear all traces (ends up like a new network)
    public function reset():Void
    {
        restore();
        
        var inputLayer:Layer = layers.input;
        var outputLayer:Layer = layers.output;
        
        inputLayer.reset();
        
        for (hiddenLayer in layers.hidden)
            hiddenLayer.reset();
            
        outputLayer.reset();
        
        if (optimized != null)
            optimized.reset();
    }
    
    // hardcodes the behaviour of the whole network into a single optimized function
    public function optimize():Dynamic
    {
        throw "Not implemented";
    }
    
    // restores all the values from the optimized network their respective
    // objects in order to manipulate the network
    public function restore():Void
    {
        throw "Not implemented";
    }
    
    // returns all the neurons in the network
    public function neurons():Array<NeuronLayerInfo>
    {
        var neurons:Array<NeuronLayerInfo> = [];
        
        var inputNeurons:Array<Neuron> = layers.input.neurons();
        var outputNeurons:Array<Neuron> = layers.output.neurons();
        
        for (n in inputNeurons)
            neurons.push({ neuron: n, layer: "input" });
        
        for (i in 0...layers.hidden.length)
        {
            var hiddenNeurons:Array<Neuron> = layers.hidden[i].neurons();
            
            for (n in hiddenNeurons)
                neurons.push({ neuron: n, layer: Std.string(i) });
        }
        
        for (n in outputNeurons)
            neurons.push({ neuron: n, layer: "output" });
        
        return neurons;
    }
    
    // returns number of inputs of the network
    public function inputs():Int
    {
        return layers.input.size;
    }
    
    // returns number of outputs of hte network
    public function outputs():Int
    {
        return layers.output.size;
    }
    
    // sets the layers of the network
    public function set(layers:NetworkLayers):Void
    {
        this.layers = layers;
        
        if (optimized != null)
            optimized.reset();
    }
    
    public function setOptimize(enabled:Bool):Void
    {
        //restore();
        
        if (optimized != null)
            optimized.reset();
            
        optimized = null;
        optimizeEnabled = enabled;
    }
    
    // returns a json that represents all the neurons and connections of the network
    public function toJson(ignoreTraces:Bool=false):NetworkJson
    {
        //restore();
        
        var list:Array<NeuronLayerInfo> = neurons();
        var neurons:Array<NeuronJson> = [];
        var connections:Array<NeuronConnectionJson> = [];
        
        // link id's to positions in the array
        var ids = new Map<Int, Int>();
        
        for (i in 0...list.length)
        {
            var neuron:Neuron = list[i].neuron;
            
            // ????
            //while (neuron.neuron)
            //    neuron = neuron.neuron;
                
            ids[neuron.ID] = i;
            
            var copy:NeuronJson =
            {
                trace:
                {
                    eligibility: new Map<Int, Float>(),
                    extended: new Map<Int, Map<Int, Float>>(),
                },
                
                state: neuron.state,
                old: neuron.old,
                activation: neuron.activation,
                bias: neuron.bias,
                layer: list[i].layer,
                squash: Activator.getName(neuron.squash),
            };
            
            neurons.push(copy);
        }
        
        // BUG in original source? neuron refers to incorrect reference
        if (!ignoreTraces)
        {
            // go through every single neuron again
            for (i in 0...neurons.length)
            {
                var copy:NeuronJson = neurons[i];
                var neuron:Neuron = list[i].neuron; // possible bugfix. get correct reference to neuron
                
                for (input in neuron.trace.eligibility.keys())
                    copy.trace.eligibility[input] = neuron.trace.eligibility[input];
                    
                for (gated in neuron.trace.extended.keys())
                {
                    copy.trace.extended[gated] = new Map<Int, Float>();
                    
                    for (input in neuron.trace.extended[gated].keys())
                        copy.trace.extended[ids[gated]][input] = neuron.trace.extended[gated][input];
                }
            }
        }
        
        // get connections
        for (i in 0...list.length)
        {
            var neuron:Neuron = list[i].neuron;
            
            //while (neuron.neuron)
            //    neuron = neuron.neuron;
            
            for (j in neuron.connections.projected.keys())
            {
                var connection:NeuronConnection = neuron.connections.projected[j];
                
                connections.push({
                    from: ids[connection.from.ID],
                    to: ids[connection.to.ID],
                    weight: connection.weight,
                    gater: connection.gater != null ? ids[connection.gater.ID] : null,
                });
            }
            
            if (neuron.isSelfConnected())
            {
                connections.push({
                    from: ids[neuron.ID],
                    to: ids[neuron.ID],
                    weight: neuron.selfconnection.weight,
                    gater: neuron.selfconnection.gater != null ? ids[neuron.selfconnection.gater.ID] : null,
                });
            }
        }
        
        return
        {
            neurons: neurons,
            connections: connections,
        }
    }
    
    // returns a function that works as the activation of the network
    // and can be used without depending on the library
    public function standalone():Dynamic
    {
        throw "Not implemented";
    }
    
    public function worker():Dynamic
    {
        throw "Not implemented";
    }
    
    // returns a copy of the network
    public function clone(ignoreTraces:Bool=false):Network
    {
        return Network.fromJson(toJson(ignoreTraces));
    }
    
    
    
    public static function fromJson(json:NetworkJson):Network
    {
        var neurons:Array<Neuron> = [];
        
        var layers:NetworkLayers =
        {
            input: new Layer(),
            hidden: [],
            output: new Layer(),
        }
        
        for (i in 0...json.neurons.length)
        {
            var config:NeuronJson = json.neurons[i];
            
            var neuron:Neuron = new Neuron();
            neuron.trace.eligibility = config.trace.eligibility;
            neuron.trace.extended = config.trace.extended;
            neuron.state = config.state;
            neuron.old = config.old;
            neuron.activation = config.activation;
            neuron.bias = config.bias;
            neuron.squash = Activator.resolve(config.squash);
            neurons.push(neuron);
            
            if (config.layer == "input")
                layers.input.add(neuron);
            else if (config.layer == 'output')
                layers.output.add(neuron);
            else
            {
                var i:Int = Std.parseInt(config.layer);
                
                if (layers.hidden.length <= i)
                    layers.hidden[i] = new Layer();
                layers.hidden[i].add(neuron);
            }
        }
        
        for (i in 0...json.connections.length)
        {
            var config:NeuronConnectionJson = json.connections[i];
            var from:Neuron = neurons[config.from];
            var to:Neuron = neurons[config.to];
            var weight:Float = config.weight;
            var gater:Neuron = config.gater != null ? neurons[config.gater] : null;
            
            var connection:NeuronConnection = from.project(to, weight);
            
            if (gater != null)
                gater.gate(connection);
        }
        
        return new Network(layers);
    }
}


interface INetwork
{
    public function activate(input:Array<Float>):Array<Float>;
    public function propagate(rate:Float=0.1, target:Array<Float>):Void;
}


typedef NetworkLayers =
{
    var input:Layer;
    var hidden:Array<Layer>;
    var output:Layer;
}

typedef NeuronLayerInfo =
{
    var neuron:Neuron;
    var layer:String;
}

typedef NeuronJsonTrace =
{
    var eligibility:Map<Int, Float>;
    var extended:Map<Int, Map<Int, Float>>;
}

typedef NetworkJson =
{
    var neurons:Array<NeuronJson>;
    var connections:Array<NeuronConnectionJson>;
}

typedef NeuronJson =
{
    var trace:NeuronJsonTrace;
    var state:Float;
    var old:Float;
    var activation:Float;
    var bias:Float;
    var layer:String;
    var squash:String;
}

typedef NeuronConnectionJson =
{
    var from:Int;
    var to:Int;
    var weight:Float;
    var gater:Null<Int>;
}

class NetworkOptimized
{
    public function activate(?input:Array<Float>):Array<Float>
    {
        throw "Not implemented";
    }
    
    public function propagate(rate:Float=0.1, ?target:Array<Float>):Void
    {
        throw "Not implemented";
    }
    
    public function reset():Void
    {
        throw "Not implemented";
    }
}
