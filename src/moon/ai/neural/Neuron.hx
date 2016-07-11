package moon.ai.neural;

import moon.ai.neural.Activator.IActivator;
import moon.ai.neural.Neuron.NeuronConnections;


/**
 * https://github.com/cazala/synaptic/wiki/Neurons
 * @author Munir Hussin
 */
class Neuron
{
    private static var LAST_ID:Int = 0;
    
    public var ID:Int;
    public var label:String = null;
    
    public var connections:NeuronConnections;
    /*{
        inputs: new Map<Int, NeuronConnection>(),
        projected: new Map<Int, NeuronConnection>(),
        gated: new Map<Int, NeuronConnection>(),
    };*/
    
    public var error:NeuronError =
    {
        responsibility: 0.0,
        projected: 0.0,
        gated: 0.0,
    };
    
    public var trace:NeuronTrace =
    {
        eligibility: new Map<Int, Float>(),
        extended: new Map<Int, Map<Int, Float>>(),
        influences: new Map<Int, Array<NeuronConnection>>(),
    };
    
    public var state:Float = 0.0;
    public var old:Float = 0.0;
    
    public var activation:Float = 0.0;
    public var derivative:Float;
    
    public var selfconnection:NeuronConnection;
    public var squash:IActivator;
    public var neighbors:Map<Int, Neuron>;
    public var bias:Float;
    
    
    public function new(?activator:IActivator) 
    {
        this.ID = uid();
        this.connections = new NeuronConnections();
        this.selfconnection = new NeuronConnection(this, this, 0.0);
        this.squash = activator == null ? Activator.init : activator;
        this.neighbors = new Map<Int, Neuron>();
        this.bias = Math.random() * 0.2 - 0.1;
    }
    
    private static inline function uid():Int
    {
        return LAST_ID++;
    }
    
    /**
     * Activate the neuron
     */
    public function activate(?input:Float):Float
    {
        if (input != null)
        {
            activation = input;
            derivative = 0.0;
            bias = 0.0;
            return activation;
        }
        
        // save old state
        old = state;
        
        // eq. 15
        state = selfconnection.gain * selfconnection.weight * state + bias;
        
        // sum the values from every input
        for (input in connections.inputs)
        {
            state += input.from.activation * input.weight * input.gain;
        }
        
        // eq. 16
        activation = squash.activation(state);
        
        // f'(s)
        derivative = squash.derivative(state);
        
        // update traces
        var influences:Array<Float> = [];
        for (id in trace.extended.keys())
        {
            // extended elegibility trace
            var xtrace:Map<Int, Float> = trace.extended[id];
            var neuron:Neuron = neighbors[id];
            
            // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
            var influence:Float = neuron.selfconnection.gater == this ? neuron.old : 0.0;
            
            // index runs over all the incoming connections to the gated neuron that are gated by this unit
            for (incoming in 0...trace.influences[neuron.ID].length)
            {
                // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
                influence += trace.influences[neuron.ID][incoming].weight *
                    trace.influences[neuron.ID][incoming].from.activation;
            }
            
            influences[neuron.ID] = influence;
        }
        
        
        for (input in connections.inputs)
        {
            // elegibility trace - Eq. 17
            trace.eligibility[input.ID] = selfconnection.gain * selfconnection.weight *
                trace.eligibility[input.ID] + input.gain * input.from.activation;
                
            for (id in trace.extended.keys())
            {
                // extended elegibility trace
                var xtrace:Map<Int, Float> = trace.extended[id];
                var neuron:Neuron = neighbors[id];
                var influence:Float = influences[neuron.ID];
                
                // eq. 18
                xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection.weight *
                    xtrace[input.ID] + derivative * trace.eligibility[input.ID] * influence;
            }
        }
        
        //  update gated connection's gains
        for (c in connections.gated)
            c.gain = activation;
            
        return activation;
    }
    
    /**
     * Back-propagate the error
     */
    public function propagate(rate:Float=0.1, ?target:Float):Void
    {
        // error accumulator
        var errorValue:Float = 0;
        
        // whether or not this neuron is in the output layer
        var isOutput = target != null;
        
        // output neurons get their error from the environment
        if (isOutput)
        {
            error.responsibility = error.projected = target - activation;
        }
        // the rest of the neuron compute their error responsibilities by backpropogation
        else
        {
            // error responsibilities from all the connections projected from this neuron
            for (connection in connections.projected)
            {
                var neuron:Neuron = connection.to;
                // Eq. 21
                errorValue += neuron.error.responsibility * connection.gain * connection.weight;
            }
            
            // projected error responsibility
            error.projected = derivative * errorValue;
            errorValue = 0;
            
            // error responsibilities from all the connections gated by this neuron
            for (id in trace.extended.keys())
            {
                // gated neuron
                var neuron:Neuron = neighbors[id];
                
                // if gated neuron's selfconnection is gated by this neuron
                var influence:Float = neuron.selfconnection.gater == this ? neuron.old : 0.0;
                
                // index runs over all the connections to the gated neuron that are gated by this neuron
                for (input in 0...trace.influences[id].length)
                {
                    // captures the effect that the input connection of this neuron have,
                    // on a neuron which its input/s is/are gated by this neuron
                    influence += trace.influences[id][input].weight *
                        trace.influences[neuron.ID][input].from.activation;
                }
                // eq. 22
                errorValue += neuron.error.responsibility * influence;
            }
            
            // gated error responsibility
            error.gated = derivative * errorValue;
            
            // error responsibility - Eq. 23
            error.responsibility = error.projected + error.gated;
        }
        
        
        // adjust all the neuron's incoming connections
        for (input in connections.inputs)
        {
            // Eq. 24
            var gradient:Float = error.projected * trace.eligibility[input.ID];
            
            for (id in trace.extended.keys())
            {
                var neuron:Neuron = neighbors[id];
                gradient += neuron.error.responsibility * trace.extended[neuron.ID][input.ID];
            }
            
            // adjust weights - aka learn
            input.weight += rate * gradient;
        }
        
        // adjust bias
        bias += rate * error.responsibility;
    }
    
    public function project(neuron:Neuron, ?weight:Float):NeuronConnection
    {
        // self-connection
        if (neuron == this)
        {
            selfconnection.weight = 1.0;
            return selfconnection;
        }
        
        // check if connection already exists
        var connected:NeuronConnectionInfo = getConnectionInfo(neuron);
        
        // connection already exist, so update weight instead if necessary
        if (connected != null && connected.type == NeuronConnectionType.Projected)
        {
            // update connection
            if (weight != null)
                connected.connection.weight = weight;
            
            // return existing connection
            return connected.connection;
        }
        
        
        // create a new connection
        var connection = new NeuronConnection(this, neuron, weight);
        
        // reference all the connections and traces
        connections.projected[connection.ID] = connection;
        neighbors[neuron.ID] = neuron;
        neuron.connections.inputs[connection.ID] = connection;
        neuron.trace.eligibility[connection.ID] = 0.0;
        
        for (trace in neuron.trace.extended)
            trace[connection.ID] = 0.0;
            
        return connection;
    }
    
    public function gate(connection:NeuronConnection):Void
    {
        // add connection to gated list
        connections.gated[connection.ID] = connection;
        
        var neuron:Neuron = connection.to;
        
        if (!trace.extended.exists(neuron.ID))
        {
            // extended trace
            neighbors[neuron.ID] = neuron;
            
            var xtrace = new Map<Int, Float>();
            trace.extended[neuron.ID] = xtrace;
            
            for (input in connections.inputs)
                xtrace[input.ID] = 0.0;
        }
        
        // keep track
        if (trace.influences.exists(neuron.ID))
            trace.influences[neuron.ID].push(connection);
        else
            trace.influences[neuron.ID] = [connection];
        
        // set gater
        connection.gater = this;
    }
    
    /**
     * Returns true or false whether the neuron is self-connected or not
     */
    public function isSelfConnected():Bool
    {
        return selfconnection.weight != 0;
    }
    
    /**
     * Returns true or false whether this neuron is connected to another neuron (parameter)
     * TODO: rename to getConnectionType
     */
    public function getConnectionInfo(neuron:Neuron):NeuronConnectionInfo
    {
        var result:NeuronConnectionInfo =
        {
            type: null,
            connection: null,
        };
        
        if (this == neuron)
        {
            if (isSelfConnected())
            {
                result.type = NeuronConnectionType.Self;
                result.connection = this.selfconnection;
                return result;
            }
            else
            {
                return null;
            }
        }
        
        // TODO: optimize
        // gated, inputs, projected
        /*for (type in Reflect.fields(connections)) // :String
        {
            for (connection in (Reflect.field(connections, type):Map<Int, NeuronConnection>))
            {
                if (connection.to == neuron || connection.from == neuron)
                {
                    result.type = type; // gated | inputs | projected
                    result.connection = connection;
                    return result;
                }
            }
        }*/
        
        for (type in NeuronConnections.types) // Inputs, Projected, Gated
        {
            for (connection in connections.get(type))
            {
                if (connection.to == neuron || connection.from == neuron)
                {
                    result.type = type;
                    result.connection = connection;
                    return result;
                }
            }
        }
        
        return null;
    }
    
    /**
     * Clears all the traces (the neuron forgets it's context, but the connections remain intact)
     */
    public function clear():Void
    {
        for (t in trace.eligibility.keys())
            trace.eligibility[t] = 0;
            
        for (t in trace.extended)
            for (e in t.keys())
                t[e] = 0.0;
        
        error.responsibility = error.projected = error.gated = 0;
    }
    
    /**
     * all the connections are randomized and the traces are cleared
     */
    public function reset():Void
    {
        clear();
        
        // gated, inputs, projected
        for (type in Reflect.fields(connections))
            for (connection in (Reflect.field(connections, type):Map<Int, NeuronConnection>))
                connection.randomizeWeight();
                
        bias = Math.random() * 0.2 - 0.1;
        old = state = activation = 0.0;
    }
    
}


class NeuronConnection
{
    private static var LAST_ID:Int = 0;
    
    public var ID:Int;
    public var from:Neuron;
    public var to:Neuron;
    public var weight:Float;
    public var gain:Float;
    public var gater:Neuron;
    
    
    public function new(from:Neuron, to:Neuron, ?weight:Float) 
    {
        if (from == null || to == null)
            throw "Invalid neurons";
            
        this.ID = uid();
        this.from = from;
        this.to = to;
        this.weight = weight == null ? getRandom() : weight;
        this.gain = 1.0;
        this.gater = null;
    }
    
    private static inline function uid():Int
    {
        return LAST_ID++;
    }
    
    public static inline function getRandom():Float
    {
        return Math.random() * 0.2 - 0.1;
    }
    
    public inline function randomizeWeight():Void
    {
        this.weight = getRandom();
    }
    
}


private typedef NeuronConnectionInfo =
{
    var type:NeuronConnectionType;
    var connection:NeuronConnection;
}

@:enum abstract NeuronConnectionType(Int) to Int from Int
{
    var Inputs      = 0;
    var Projected   = 1;
    var Gated       = 2;
    var Self        = 3;
}

/*private typedef NeuronConnections =
{
    var inputs:Map<Int, NeuronConnection>;
    var projected:Map<Int, NeuronConnection>;
    var gated:Map<Int, NeuronConnection>;
}*/


class NeuronConnections
{
    public static var types:Array<NeuronConnectionType> = [Inputs, Projected, Gated];
    
    private var connections:Array<Map<Int, NeuronConnection>>;
    public var inputs(get, never):Map<Int, NeuronConnection>;
    public var projected(get, never):Map<Int, NeuronConnection>;
    public var gated(get, never):Map<Int, NeuronConnection>;
    
    public function new()
    {
        this.connections =
        [
            new Map<Int, NeuronConnection>(),
            new Map<Int, NeuronConnection>(),
            new Map<Int, NeuronConnection>(),
        ];
    }
    
    public inline function get(type:NeuronConnectionType):Map<Int, NeuronConnection>
    {
        return connections[type];
    }
    
    private inline function get_inputs():Map<Int, NeuronConnection>
    {
        return connections[NeuronConnectionType.Inputs];
    }
    
    private inline function get_projected():Map<Int, NeuronConnection>
    {
        return connections[NeuronConnectionType.Projected];
    }
    
    private inline function get_gated():Map<Int, NeuronConnection>
    {
        return connections[NeuronConnectionType.Gated];
    }
}


private typedef NeuronError =
{
    var responsibility:Float;
    var projected:Float;
    var gated:Float;
}

private typedef NeuronTrace =
{
    var eligibility:Map<Int, Float>;
    var extended:Map<Int, Map<Int, Float>>;
    var influences:Map<Int, Array<NeuronConnection>>;
}





