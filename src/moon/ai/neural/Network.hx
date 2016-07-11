package moon.ai.neural;

import moon.ai.neural.Layer.ConnectionType;
import moon.ai.neural.Layer.GateType;
import moon.ai.neural.Layer.ILayerProjectable;
import moon.ai.neural.Layer.LayerConnection;
import moon.ai.neural.Network.NetworkLayers;
import moon.ai.neural.Neuron.NeuronConnection;

using StringTools;

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
    
    
    /**
     * Feed-forward activation of all the layers to produce an ouput
     */
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
    
    /**
     * Back-propagate the error thru the network
     */
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
    
    /**
     * Project a connection to another unit (either a network or a layer)
     */
    public function project(unit:ILayerProjectable, ?type:ConnectionType, ?weight:Float):LayerConnection
    {
        if (optimized != null)
            optimized.reset();
            
        return layers.output.project(unit.getProjectableLayer(), type, weight);
    }
    
    /**
     * Interface method to get a projectable layer
     */
    public function getProjectableLayer():Layer
    {
        return layers.input;
    }
    
    /**
     * Let this network gate a connection
     */
    public function gate(connection:LayerConnection, type:GateType):Void
    {
        if (optimized != null)
            optimized.reset();
            
        layers.output.gate(connection, type);
    }
    
    /**
     * clear all elegibility traces and extended elegibility traces
     * (the network forgets its context, but not what was trained)
     */
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
    
    /**
     * Reset all weights and clear all traces (ends up like a new network)
     */
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
    
    /**
     * (NOT IMPLEMENTED - can't dynamically generate functions in all targets)
     * Hardcodes the behaviour of the whole network into a single optimized function
     */
    public function optimize():Dynamic
    {
        throw "Not implemented";
    }
    
    /**
     * (NOT IMPLEMENTED)
     * Restores all the values from the optimized network their respective
     * objects in order to manipulate the network
     */
    public function restore():Void
    {
        throw "Not implemented";
    }
    
    /**
     * Returns all the neurons in the network
     */
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
    
    /**
     * Returns number of inputs of the network
     */
    public function inputs():Int
    {
        return layers.input.size;
    }
    
    /**
     * Returns number of outputs of the network
     */
    public function outputs():Int
    {
        return layers.output.size;
    }
    
    /**
     * Sets the layers of the network
     */
    public function set(layers:NetworkLayers):Void
    {
        this.layers = layers;
        
        if (optimized != null)
            optimized.reset();
    }
    
    /**
     * Enabling optimize doesn't work because dynamically generating functions is not
     * supported in all haxe targets
     */
    public function setOptimize(enabled:Bool):Void
    {
        //restore(); // throws error
        
        if (optimized != null)
            optimized.reset();
            
        optimized = null;
        optimizeEnabled = enabled;
    }
    
    /**
     * Returns a json that represents all the neurons and connections of the network
     */
    public function toJson(ignoreTraces:Bool=false):NetworkJson
    {
        //restore(); // throws error
        
        var list:Array<NeuronLayerInfo> = neurons();
        var neurons:Array<NeuronJson> = [];
        var connections:Array<NeuronConnectionJson> = [];
        
        // link id's to positions in the array
        var ids = new Map<Int, Int>();
        
        for (i in 0...list.length)
        {
            var neuron:Neuron = list[i].neuron;
            
            // COMPILE ERROR: moon.ai.neural.Neuron has no field neuron
            // possible mistake in original source?
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
        
        // no longer in original source
        /*// BUG in original source? neuron refers to incorrect reference
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
        }*/
        
        // get connections
        for (i in 0...list.length)
        {
            var neuron:Neuron = list[i].neuron;
            
            // COMPILE ERROR: moon.ai.neural.Neuron has no field neuron
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
    
    /**
     * Export the topology into dot language which can be visualized as graphs using dot.
     * example: ... Sys.println(net.toDot());
     *     $ neko example.n > example.dot
     *     $ dot example.dot -Tpng > out.png
     */
    public function toDot(edgeConnection:Bool=false)
    {
        var code:String = "digraph nn {\n    rankdir = BT\n";
        var layers:Array<Layer> = [this.layers.input].concat(this.layers.hidden.concat([this.layers.output]));
        
        for (layerID in 0...layers.length)
        {
            var layer:Layer = layers[layerID];
            
            // projections
            for (connection in layer.connectedTo)
            {
                var layerTo:Layer = connection.to;
                var size:Int = connection.size;
                //var layerID:Int = layers.indexOf(layers[layer]); // OPTIMIZED: doesn't need to be computed every loop. same as layerID above
                var layerToID:Int = layers.indexOf(layerTo);
                
                // http://stackoverflow.com/questions/26845540/connect-edges-with-graph-dot
                // DOT does not support edge-to-edge connections
                // This workaround produces somewhat weird graphs ...
                
                if (edgeConnection)
                {
                    var fakeNode:String = null; // is this correct?
                    
                    if (connection.gatedFrom.length > 0)
                    {
                        fakeNode = "fake" + layerID + "_" + layerToID;
                        code += "    " + fakeNode + " [label = \"\", shape = point, width = 0.01, height = 0.01]\n";
                        code += "    " + layerID + " -> " + fakeNode + " [label = " + size + ", arrowhead = none]\n";
                        code += "    " + fakeNode + " -> " + layerToID + "\n";
                    }
                    else
                    {
                        code += "    " + layerID + " -> " + layerToID + " [label = " + size + "]\n";
                    }
                    
                    // gatings
                    for (gatedInfo in connection.gatedFrom)
                    {
                        var layerfrom = gatedInfo.layer;
                        var type = gatedInfo.type;
                        var layerfromID = layers.indexOf(layerfrom);
                        code += "    " + layerfromID + " -> " + fakeNode + " [color = blue]\n";
                    }
                }
                else
                {
                    code += "    " + layerID + " -> " + layerToID + " [label = " + size + "]\n";
                    
                    // gatings
                    for (gatedInfo in connection.gatedFrom)
                    {
                        var layerfrom = gatedInfo.layer;
                        var type = gatedInfo.type;
                        var layerfromID = layers.indexOf(layerfrom);
                        code += "    " + layerfromID + " -> " + layerToID + " [color = blue]\n";
                    }
                }
            }
        }
        
        code += "}\n";
        
        return
        {
            code: code,
            link: "https://chart.googleapis.com/chart?chl=" + code.replace("/ /g", "+").urlEncode() + "&cht=gv"
        }
    }
    
    /**
     * (NOT IMPLEMENTED)
     * Returns a function that works as the activation of the network
     * and can be used without depending on the library
     */
    public function standalone():Dynamic
    {
        throw "Not implemented";
    }
    
    /**
     * (NOT IMPLEMENTED)
     * Return a HTML5 WebWorker specialized on training the network stored in `memory`.
     * Train based on the given dataSet and options.
     * The worker returns the updated `memory` when done.
     */
    public function worker(?memory:Dynamic, ?set:Dynamic, ?options:Dynamic):Dynamic
    {
        throw "Not implemented";
    }
    
    /**
     * Returns a copy of the network
     */
    public function clone(ignoreTraces:Bool=false):Network
    {
        return Network.fromJson(toJson(ignoreTraces));
    }
    
    
    /**
     * Rebuild a network that has been stored in a json using the method toJSON()
     */
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
                
                if (i >= layers.hidden.length)
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
