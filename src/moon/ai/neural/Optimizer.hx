package moon.ai.neural;

// for network
import moon.ai.neural.Network.NeuronLayerInfo;

#if sys
    import sys.io.File;
#end

// for neuron
import moon.core.Struct;
import moon.ai.neural.activators.HardLimit;
import moon.ai.neural.activators.Identity;
import moon.ai.neural.activators.Logistic;
import moon.ai.neural.activators.Tanh;
import moon.ai.neural.Neuron.NeuronConnection;

/**
 * ...
 * @author Munir Hussin
 */
class Optimizer
{
    public var that:Network;
    
    public function new(that:Network) 
    {
        this.that = that;
    }
    
    #if sys
    // save(network, "../../src", "moonaire.util.ai.neural.generated.Blah")
    public static function save(that:Network, pathToSrc:String, qualifiedId:String):Void
    {
        // ensure trailing slash
        if (pathToSrc.charAt(pathToSrc.length - 1) != "/")
            pathToSrc += "/";
            
        var parts:Array<String> = qualifiedId.split(".");
        var cls:String = parts.pop();
        var pkg:String = parts.join(".");
        var path:String = pathToSrc + parts.join("/") + "/" + cls + ".hx";
        
        var codes:String = optimize(that, pkg, cls);
        File.saveContent(path, codes);
    }
    #end
    
    public static function optimize(that:Network, pkg:String, cls:String):String
    {
        return new Optimizer(that).run(pkg, cls);
    }
    
    public function template(pkg:String, cls:String, memory:Int, init:String, activate:String, propagate:String):String
    {
        return
        [
            'package $pkg;',
            '',
            'import haxe.ds.Vector;',
            'import moon.ai.neural.Network.INetwork;',
            '',
            'class $cls implements INetwork',
            '{',
            '    public var f:Vector<Float>;',
            '',
            '    public function new()',
            '    {',
            '        f = new Vector<Float>($memory);',
            init,
            '    }',
            '    ',
            '    public function activate(input:Array<Float>):Array<Float>',
            '    {',
            activate,
            '    }',
            '    ',
            '    public function propagate(rate:Float=0.1, target:Array<Float>):Void',
            '    {',
            propagate,
            '    }',
            '    ',
            '    public function ownership(memory:Vector<Float>):Void',
            '    {',
            '        f = memory;',
            '    }',
            '}',
            '',
        ].join("\n");
    }
    
    public function run(pkg:String, cls:String):String
    {
        var optimized:Optimized = null;
        var neurons:Array<NeuronLayerInfo> = that.neurons();
        
        for (i in 0...neurons.length)
        {
            var neuron:Neuron = neurons[i].neuron;
            var layer:String = neurons[i].layer;
            
            // ????
            //while (neuron.neuron)
            //    neuron = neuron.neuron;
            
            optimized = NeuronOptimizer.optimize(neuron, optimized, layer);
        }
        
        for (i in 0...optimized.propagation_sentences.length)
            optimized.propagation_sentences[i].reverse();
        optimized.propagation_sentences.reverse();
        
        
        var hardcode:String = "";
        var initCodes:Array<String> = [];
        var activateCodes:Array<String> = [];
        var propagateCodes:Array<String> = [];
        
        for (variable in optimized.variables)
            initCodes.push("        f[" + variable.id + "] = " + (variable.value != null ? variable.value : 0) + ";");
            
        // activate function
        
        for (i in 0...optimized.inputs.length)
            activateCodes.push("        f[" + optimized.inputs[i] + "] = input[" + i + "];");
            
        for (currentLayer in 0...optimized.activation_sentences.length)
        {
            if (optimized.activation_sentences[currentLayer].length > 0)
            {
                for (currentNeuron in 0...optimized.activation_sentences[currentLayer].length)
                {
                    activateCodes.push(optimized.activation_sentences[currentLayer][currentNeuron].join("\n"));
                    activateCodes.push(optimized.trace_sentences[currentLayer][currentNeuron].join("\n"));
                }
            }
        }
        
        activateCodes.push("        ");
        activateCodes.push("        var output:Array<Float> = [];");
        
        for (i in 0...optimized.outputs.length)
            activateCodes.push("        output[" + i + "] = f[" + optimized.outputs[i] + "];");
            
        activateCodes.push("        return output;");
        
        // propagate function
        propagateCodes.push("        f[" + optimized.variables.get("rate").id + "] = rate;");
        
        for (i in 0...optimized.targets.length)
            propagateCodes.push("        f[" + optimized.targets[i] + "] = target[" + i + "];");
            
        for (currentLayer in 0...optimized.propagation_sentences.length)
            for (currentNeuron in 0...optimized.propagation_sentences[currentLayer].length)
                propagateCodes.push(optimized.propagation_sentences[currentLayer][currentNeuron].join("\n"));
                
        return template(pkg, cls, optimized.memory, initCodes.join("\n"),
            activateCodes.join("\n"), propagateCodes.join("\n"));
        
        /*var constructor = new Function(hardcode);
        
        var network = constructor();
        network.data = {
            variables: optimized.variables,
            activate: optimized.activation_sentences,
            propagate: optimized.propagation_sentences,
            trace: optimized.trace_sentences,
            inputs: optimized.inputs,
            outputs: optimized.outputs,
            check_activation: this.activate,
            check_propagation: this.propagate
        }

        network.reset = function() {
            if (that.optimized) {
                that.optimized = null;
                that.activate = network.data.check_activation;
                that.propagate = network.data.check_propagation;
            }
        }

        this.optimized = network;
        this.activate = network.activate;
        this.propagate = network.propagate;*/
    }
    
}










class NeuronOptimizer
{
    public var that:Neuron;
    public var layer:String;
    
    public var store_activation:Array<String>;
    public var store_trace:Array<String>;
    public var store_propagation:Array<String>;
    
    public var varID:Int;
    public var neurons:Int;
    public var inputs:Array<String>;
    public var targets:Array<Int>;
    public var outputs:Array<String>;
    public var variables:Map<String, Variable>;
    public var activation_sentences:Array<Array<Array<String>>>;
    public var trace_sentences:Array<Array<Array<String>>>;
    public var propagation_sentences:Array<Array<Array<String>>>;
    public var layers:Map<String, Int>;
    
    
    public function new(that:Neuron, optimized:Optimized, layer:String) 
    {
        optimized = Struct.options(optimized,
        {
            memory: 0,
            neurons: 1,
            inputs: [],
            targets: [],
            outputs: [],
            variables: new Map<String, Variable>(),
            activation_sentences: [],
            trace_sentences: [],
            propagation_sentences: [],
            layers:
            {
                var x = new Map<String, Int>();
                x["__count"] = 0;
                x["__neuron"] = 0;
                x;
            },
        });
        
        this.that = that;
        this.layer = layer;
        
        store_activation = [];
        store_trace = [];
        store_propagation = [];
        
        varID = optimized.memory;
        neurons = optimized.neurons;
        inputs = optimized.inputs;
        targets = optimized.targets;
        outputs = optimized.outputs;
        variables = optimized.variables;
        activation_sentences = optimized.activation_sentences;
        trace_sentences = optimized.trace_sentences;
        propagation_sentences = optimized.propagation_sentences;
        layers = optimized.layers;
    }
    
    
    public static function optimize(that:Neuron, optimized:Optimized, layer:String):Optimized
    {
        return new NeuronOptimizer(that, optimized, layer).run();
    }
    
    
    // allocate sentences
    public function allocate(store:Array<Dynamic>):Void
    {
        //var allocated = layer in layers && store[layers.__count];
        var allocated:Bool = layers.exists(layer) && store[layers["__count"]];
        
        if (!allocated)
        {
            layers["__count"] = store.push([]) - 1;
            layers[layer] = layers["__count"];
        }
    }
    
    // get/reserve space in memory by creating a unique ID for a variable
    // targets:Array<Int>
    public function getVar(args:Array<Dynamic>):Variable
    {
        var id:String = null;
        var value:Float = 0;
        
        // getVar(x)
        if (args.length == 1)
        {
            if (args[0] == 'target')
            {
                id = 'target_' + targets.length;
                targets.push(varID);
            }
            else
            {
                id = args[0];
            }
        }
        // extended:  getVar(unit, prop1, prop2, prop3, value)
        // !extended: getVar(unit, prop) ==> value = unit[prop]
        else
        {
            
            var extended:Bool = args.length > 2;
            
            if (extended)
                value = args.pop();
                
            var unit = args.shift();
            var prop = args.pop();
            
            if (!extended)
                value = Reflect.field(unit, prop);
                
            id = prop + '_';
            
            for (property in args)
                id += property + '_';
            
            id += unit.ID;
        }
        
        
        if (variables.exists(id))
            return variables[id];
            
        return variables[id] = cast
        {
            value: value,
            id: varID++
        };
    }
    
    // build sentence
    // buildSentence([a, b, c], storage)
    public function buildSentence(args:Array<Dynamic>, store:Array<String>):Void
    {
        var sentence:String = "        ";
        
        for (i in 0...args.length)
            if (Std.is(args[i], String)) // String or Variable
                sentence += args[i];
            else
                sentence += 'f[' + args[i].id + ']';
                
        store.push(sentence + ';');
    }
    
    // helper to check if an object is empty
    public function isEmpty(obj:Dynamic):Bool
    {
        return Reflect.fields(obj).length == 0;
    }
    
    public function run():Optimized
    {
        allocate(activation_sentences);
        allocate(trace_sentences);
        allocate(propagation_sentences);
        
        var currentLayer:Int = layers["__count"];
        
        
        // characteristics of the neuron
        var noProjections:Bool = isEmpty(that.connections.projected);
        var noGates:Bool = isEmpty(that.connections.gated);
        var isInput:Bool = layer == "input" ? true : isEmpty(that.connections.inputs);
        var isOutput:Bool = layer == "output" ? true : noProjections && noGates;
        
        // optimize neuron's behaviour
        var rate:Variable = getVar(['rate']);
        var activation:Variable = getVar([that, 'activation']);
        var derivative:Variable = getVar([that, 'derivative']);
        var old:Variable = getVar([that, 'old']);
        var state:Variable = getVar([that, 'state']);
        var bias:Variable = getVar([that, 'bias']);
        
        if (isInput)
        {
            inputs.push(activation.id);
        }
        else
        {
            activation_sentences[currentLayer].push(store_activation);
            trace_sentences[currentLayer].push(store_trace);
            propagation_sentences[currentLayer].push(store_propagation);
            
            
            var self_gain:Variable = null;
            var self_weight:Variable = null;
            
            buildSentence([old, ' = ', state], store_activation);
            
            if (that.isSelfConnected())
            {
                self_weight = getVar([that.selfconnection, 'weight']);
                
                if (that.selfconnection.gater != null)
                {
                    self_gain = getVar([that.selfconnection, 'gain']);
                    buildSentence([state, ' = ', self_gain, ' * ', self_weight, ' * ', state, ' + ', bias], store_activation);
                }
                else
                {
                    buildSentence([state, ' = ', self_weight, ' * ', state, ' + ', bias], store_activation);
                }
            }
            else
            {
                buildSentence([state, ' = ', bias], store_activation);
            }
            
            for (i in that.connections.inputs.keys())
            {
                var input:NeuronConnection = that.connections.inputs[i];
                var input_activation:Variable = getVar([input.from, 'activation']);
                var input_weight:Variable = getVar([input, 'weight']);
                
                if (input.gater != null)
                {
                    var input_gain:Variable = getVar([input, 'gain']);
                    buildSentence([state, ' += ', input_activation, ' * ', input_weight, ' * ', input_gain], store_activation);
                }
                else
                {
                    buildSentence([state, ' += ', input_activation, ' * ', input_weight], store_activation);
                }
            }
            
            
            
            switch (Type.getClass(that.squash))
            {
                case Logistic:
                    buildSentence([activation, ' = (1.0 / (1.0 + Math.exp(-', state, ')))'], store_activation);
                    buildSentence([derivative, ' = ', activation, ' * (1.0 - ', activation, ')'], store_activation);
                        
                case Tanh:
                    var eP:Variable = getVar(['aux']);
                    var eN:Variable = getVar(['aux_2']);
                    buildSentence([eP, ' = Math.exp(', state, ')'], store_activation);
                    buildSentence([eN, ' = 1.0 / ', eP], store_activation);
                    buildSentence([activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')'], store_activation);
                    buildSentence([derivative, ' = 1.0 - (', activation, ' * ', activation, ')'], store_activation);
                    
                case Identity:
                    buildSentence([activation, ' = ', state], store_activation);
                    buildSentence([derivative, ' = 1.0'], store_activation);
                    
                case HardLimit:
                    buildSentence([activation, ' = (', state, ' > 0) ? 1.0 : 0.0'], store_activation);
                    buildSentence([derivative, ' = 1.0'], store_activation);
            }
            
            for (i in that.connections.inputs.keys())
            {
                var input:NeuronConnection = that.connections.inputs[i];
                var input_gain:Variable = null;
                var input_activation:Variable = getVar([input.from, 'activation']);
                var trace:Variable = getVar([that, 'trace', 'elegibility', input.ID, that.trace.eligibility[input.ID]]);
                
                if (input.gater != null)
                    input_gain = getVar([input, 'gain']);
                    
                if (that.isSelfConnected())
                {
                    if (that.selfconnection.gater != null)
                    {
                        if (input.gater != null)
                            buildSentence([trace, ' = ', self_gain, ' * ', self_weight, ' * ',
                                trace, ' + ', input_gain, ' * ', input_activation], store_trace);
                        else
                            buildSentence([trace, ' = ', self_gain, ' * ', self_weight, ' * ',
                                trace, ' + ', input_activation], store_trace);
                    }
                    else
                    {
                        if (input.gater != null)
                            buildSentence([trace, ' = ', self_weight, ' * ', trace, ' + ',
                                input_gain, ' * ', input_activation], store_trace);
                        else
                            buildSentence([trace, ' = ', self_weight, ' * ', trace, ' + ',
                                input_activation], store_trace);
                    }
                }
                else
                {
                    if (input.gater != null)
                        buildSentence([trace, ' = ', input_gain, ' * ', input_activation], store_trace);
                    else
                        buildSentence([trace, ' = ', input_activation], store_trace);
                }
                
                for (id in that.trace.extended.keys())
                {
                    // extended elegibility trace
                    var xtrace:Map<Int, Float> = that.trace.extended[id];
                    var neuron:Neuron = that.neighbors[id];
                    var influence:Variable = getVar(['aux']);
                    var neuron_old:Variable = getVar([neuron, 'old']);
                    
                    if (neuron.selfconnection.gater == that)
                        buildSentence([influence, ' = ', neuron_old], store_trace);
                    else
                        buildSentence([influence, ' = 0'], store_trace);
                        
                    for (incoming in that.trace.influences[neuron.ID])
                    {
                        var incoming_weight:Variable = getVar([incoming, 'weight']);
                        var incoming_activation:Variable = getVar([incoming.from, 'activation']);
                        
                        buildSentence([influence, ' += ', incoming_weight, ' * ', incoming_activation], store_trace);
                    }
                    
                    var trace:Variable = getVar([that, 'trace', 'elegibility', input.ID,
                        that.trace.eligibility[input.ID]]);
                    var xtrace:Variable = getVar([that, 'trace', 'extended', neuron.ID, input.ID,
                        that.trace.extended[neuron.ID][input.ID]]);
                    
                    
                    if (neuron.isSelfConnected())
                    {
                        var neuron_self_weight:Variable = getVar([neuron.selfconnection, 'weight']);
                        
                        if (neuron.selfconnection.gater != null)
                        {
                            var neuron_self_gain:Variable = getVar([neuron.selfconnection, 'gain']);
                            buildSentence([xtrace, ' = ', neuron_self_gain, ' * ', neuron_self_weight, ' * ',
                                xtrace, ' + ', derivative, ' * ', trace, ' * ', influence], store_trace);
                        }
                        else
                        {
                            buildSentence([xtrace, ' = ', neuron_self_weight, ' * ', xtrace, ' + ',
                                derivative, ' * ', trace, ' * ', influence], store_trace);
                        }
                    }
                    else
                    {
                        buildSentence([xtrace, ' = ', derivative, ' * ', trace, ' * ', influence], store_trace);
                    }
                }
            }
            
            for (connection in that.connections.gated)
            {
                var gated_gain:Variable = getVar([connection, 'gain']);
                buildSentence([gated_gain, ' = ', activation], store_activation);
            }
        }
        
        if (!isInput)
        {
            var responsibility:Variable = getVar([that, 'error', 'responsibility', that.error.responsibility]);
            
            if (isOutput)
            {
                var target:Variable = getVar(['target']);
                buildSentence([responsibility, ' = ', target, ' - ', activation], store_propagation);
                
                for (input in that.connections.inputs)
                {
                    var trace:Variable = getVar([that, 'trace', 'elegibility', input.ID, that.trace.eligibility[input.ID]]);
                    var input_weight:Variable = getVar([input, 'weight']);
                    
                    buildSentence([input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')'], store_propagation);
                }
                
                outputs.push(activation.id);
            }
            else
            {
                if (!noProjections && !noGates)
                {
                    var error:Variable = getVar(['aux']);
                    
                    for (id in that.connections.projected.keys())
                    {
                        var connection:NeuronConnection = that.connections.projected[id];
                        var neuron:Neuron = connection.to;
                        var connection_weight:Variable = getVar([connection, 'weight']);
                        var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                        
                        if (connection.gater != null)
                        {
                            var connection_gain:Variable = getVar([connection, 'gain']);
                            buildSentence([error, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ',
                                connection_weight], store_propagation);
                        }
                        else
                        {
                            buildSentence([error, ' += ', neuron_responsibility, ' * ', connection_weight], store_propagation);
                        }
                    }
                    
                    var projected:Variable = getVar([that, 'error', 'projected', that.error.projected]);
                    buildSentence([projected, ' = ', derivative, ' * ', error], store_propagation);
                    buildSentence([error, ' = 0'], store_propagation);
                    
                    for (id in that.trace.extended.keys())
                    {
                        var neuron = that.neighbors[id];
                        var influence:Variable = getVar(['aux_2']);
                        var neuron_old:Variable = getVar([neuron, 'old']);
                        
                        if (neuron.selfconnection.gater == that)
                            buildSentence([influence, ' = ', neuron_old], store_propagation);
                        else
                            buildSentence([influence, ' = 0'], store_propagation);
                            
                        for (connection in that.trace.influences[neuron.ID])
                        {
                            //var connection = that.trace.influences[neuron.ID][input];
                            var connection_weight:Variable = getVar([connection, 'weight']);
                            var neuron_activation:Variable = getVar([connection.from, 'activation']);
                            buildSentence([influence, ' += ', connection_weight, ' * ', neuron_activation], store_propagation);
                        }
                        
                        var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                        buildSentence([error, ' += ', neuron_responsibility, ' * ', influence], store_propagation);
                    }
                    
                    var gated:Variable = getVar([that, 'error', 'gated', that.error.gated]);
                    buildSentence([gated, ' = ', derivative, ' * ', error], store_propagation);
                    buildSentence([responsibility, ' = ', projected, ' + ', gated], store_propagation);
                    
                    for (id in that.connections.inputs.keys())
                    {
                        var input:NeuronConnection = that.connections.inputs[id];
                        var gradient:Variable = getVar(['aux']);
                        var trace:Variable = getVar([that, 'trace', 'elegibility', input.ID, that.trace.eligibility[input.ID]]);
                        buildSentence([gradient, ' = ', projected, ' * ', trace], store_propagation);
                        
                        for (id in that.trace.extended.keys())
                        {
                            var neuron:Neuron = that.neighbors[id];
                            var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                            var xtrace:Variable = getVar([that, 'trace', 'extended', neuron.ID, input.ID, that.trace.extended[neuron.ID][input.ID]]);
                            buildSentence([gradient, ' += ', neuron_responsibility, ' * ', xtrace], store_propagation);
                        }
                        
                        var input_weight:Variable = getVar([input, 'weight']);
                        buildSentence([input_weight, ' += ', rate, ' * ', gradient], store_propagation);
                    }
                    
                }
                else if (noGates)
                {
                    buildSentence([responsibility, ' = 0'], store_propagation);
                    
                    for (id in that.connections.projected.keys())
                    {
                        var connection:NeuronConnection = that.connections.projected[id];
                        var neuron:Neuron = connection.to;
                        var connection_weight:Variable = getVar([connection, 'weight']);
                        var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                        
                        if (connection.gater != null)
                        {
                            var connection_gain:Variable = getVar([connection, 'gain']);
                            buildSentence([responsibility, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ', connection_weight], store_propagation);
                        }
                        else
                        {
                            buildSentence([responsibility, ' += ', neuron_responsibility, ' * ', connection_weight], store_propagation);
                        }
                    }
                    
                    buildSentence([responsibility, ' *= ', derivative], store_propagation);
                    
                    for (id in that.connections.inputs.keys())
                    {
                        var input:NeuronConnection = that.connections.inputs[id];
                        var trace:Variable = getVar([that, 'trace', 'elegibility', input.ID, that.trace.eligibility[input.ID]]);
                        var input_weight:Variable = getVar([input, 'weight']);
                        buildSentence([input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')'], store_propagation);
                    }
                }
                else if (noProjections)
                {
                    buildSentence([responsibility, ' = 0'], store_propagation);
                    
                    for (id in that.trace.extended.keys())
                    {
                        var neuron:Neuron = that.neighbors[id];
                        var influence:Variable = getVar(['aux']);
                        var neuron_old:Variable = getVar([neuron, 'old']);
                        
                        if (neuron.selfconnection.gater == that)
                            buildSentence([influence, ' = ', neuron_old], store_propagation);
                        else
                            buildSentence([influence, ' = 0'], store_propagation);
                            
                        for (connection in that.trace.influences[neuron.ID])
                        {
                            //var connection = that.trace.influences[neuron.ID][input];
                            var connection_weight:Variable = getVar([connection, 'weight']);
                            var neuron_activation:Variable = getVar([connection.from, 'activation']);
                            buildSentence([influence, ' += ', connection_weight, ' * ', neuron_activation], store_propagation);
                        }
                        
                        var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                        buildSentence([responsibility, ' += ', neuron_responsibility, ' * ', influence], store_propagation);
                    }
                    
                    buildSentence([responsibility, ' *= ', derivative], store_propagation);
                    
                    for (id in that.connections.inputs.keys())
                    {
                        var input:NeuronConnection = that.connections.inputs[id];
                        var gradient:Variable = getVar(['aux']);
                        buildSentence([gradient, ' = 0'], store_propagation);
                        
                        for (id in that.trace.extended.keys())
                        {
                            var neuron:Neuron = that.neighbors[id];
                            var neuron_responsibility:Variable = getVar([neuron, 'error', 'responsibility', neuron.error.responsibility]);
                            var xtrace:Variable = getVar([that, 'trace', 'extended', neuron.ID, input.ID, that.trace.extended[neuron.ID][input.ID]]);
                            buildSentence([gradient, ' += ', neuron_responsibility, ' * ', xtrace], store_propagation);
                        }
                        
                        var input_weight:Variable = getVar([input, 'weight']);
                        buildSentence([input_weight, ' += ', rate, ' * ', gradient], store_propagation);
                    }
                }
            }
            
            buildSentence([bias, ' += ', rate, ' * ', responsibility], store_propagation);
        }
        
        return
        {
            memory: varID,
            neurons: neurons + 1,
            inputs: inputs,
            outputs: outputs,
            targets: targets,
            variables: variables,
            activation_sentences: activation_sentences,
            trace_sentences: trace_sentences,
            propagation_sentences: propagation_sentences,
            layers: layers
        }
    }
    
}



typedef Optimized =
{
    var memory:Int;
    var neurons:Int;
    var inputs:Array<String>;
    var outputs:Array<String>;
    var targets:Array<Int>;
    var variables:Map<String, Variable>;
    var activation_sentences:Array<Array<Array<String>>>;
    var trace_sentences:Array<Array<Array<String>>>;
    var propagation_sentences:Array<Array<Array<String>>>;
    var layers:Map<String, Int>;
}

typedef Variable =
{
    var value:Float;
    var id:String;
}



