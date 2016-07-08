package moon.ai.neural;

import moon.ai.neural.Activator.IActivator;
import moon.ai.neural.Neuron.NeuronConnection;

/**
 * A layer consist of multiple neurons.
 * A layer can connect to other layers to form a network.
 * 
 * @author Munir Hussin
 */
class Layer implements ILayerProjectable
{
    public var size:Int;
    public var list:Array<Neuron>;
    public var label:String;
    
    
    public function new(size:Int=0, ?label:String)
    {
        this.size = size;
        this.list = [];
        this.label = label;
        
        while (size-->0)
        {
            var neuron:Neuron = new Neuron();
            list.push(neuron);
        }
    }
    
    /**
     * Activates all the neurons in the layer
     */
    public function activate(?input:Array<Float>):Array<Float>
    {
        var activations:Array<Float> = [];
        
        // input is given, so activate each neuron with their respective inputs
        if (input != null)
        {
            if (input.length != size)
                throw "INPUT size and LAYER size must be the same to activate!";
                
            for (i in 0...list.length)
            {
                var neuron:Neuron = list[i];
                var activation:Float = neuron.activate(input[i]);
                activations.push(activation);
            }
        }
        else
        {
            for (neuron in list)
            {
                var activation:Float = neuron.activate();
                activations.push(activation);
            }
        }
        
        return activations;
    }
    
    /**
     * Propagates the error on all the neurons of the layer
     */
    public function propagate(rate:Float=0.1, ?target:Array<Float>):Void
    {
        if (target != null)
        {
            if (target.length != size)
                throw "TARGET size and LAYER size must be the same to propagate!";
            
            var i:Int = list.length;
            
            while (i-->0)
            {
                var neuron:Neuron = list[i];
                neuron.propagate(rate, target[i]);
            }
        }
        else
        {
            var i:Int = list.length;
            
            while (i-->0)
            {
                var neuron:Neuron = list[i];
                neuron.propagate(rate);
            }
        }
    }
    
    /**
     * Projects a connection from this layer to another one.
     * Unit can be either a network or a layer.
     */
    public function project(unit:ILayerProjectable, ?type:ConnectionType, ?weight:Float):LayerConnection
    {
        var layer:Layer = unit.getProjectableLayer();
        
        if (isConnected(layer) == null)
            return new LayerConnection(this, layer, type, weight);
            
        throw "Layer has already been connected!";
    }
    
    /**
     * Interface method to get a projectable layer.
     */
    public function getProjectableLayer():Layer
    {
        return this;
    }
    
    /**
     * Gates a connection betwenn two layers.
     */
    public function gate(connection:LayerConnection, type:GateType):Void
    {
        if (type == GateType.Input)
        {
            if (connection.to.size != size)
                throw "GATER layer and CONNECTION.TO layer must be the same size in order to gate!";
                
            for (i in 0...connection.to.list.length)
            {
                var neuron:Neuron = connection.to.list[i];
                var gater:Neuron = list[i];
                
                for (gated in neuron.connections.inputs)
                    if (connection.connections.exists(gated.ID))
                        gater.gate(gated);
            }
        }
        else if (type == GateType.Output)
        {
            if (connection.from.size != size)
                throw "GATER layer and CONNECTION.FROM layer must be the same size in order to gate!";
                
            for (i in 0...connection.from.list.length)
            {
                var neuron:Neuron = connection.from.list[i];
                var gater:Neuron = this.list[i];
                
                for (gated in neuron.connections.projected)
                    if (connection.connections.exists(gated.ID))
                        gater.gate(gated);
            }
        }
        else if (type == GateType.OneToOne)
        {
            if (connection.size != size)
                throw "The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!";
                
            for (i in 0...connection.list.length)
            {
                var gater:Neuron = list[i];
                var gated:NeuronConnection = connection.list[i];
                gater.gate(gated);
            }
        }
    }
    
    /**
     * True or false whether the whole layer is self-connected or not.
     */
    public function isSelfConnected():Bool
    {
        for (neuron in list)
            if (!neuron.isSelfConnected())
                return false;
        return true;
    }
    
    /**
     * True or false whether the layer is connected to another layer (parameter) or not.
     */
    public function isConnected(layer:Layer):ConnectionType
    {
        // Check if ALL to ALL connection
        var connections:Int = 0;
        
        for (from in list)
        {
            for (to in layer.list)
            {
                var connected = from.isConnected(to);
                
                if (connected != null && connected.type == 'projected')
                    connections++;
            }
        }
        
        if (connections == size * layer.size)
            return ConnectionType.AllToAll;

        // Check if ONE to ONE connection
        connections = 0;
        
        for (i in 0...list.length)
        {
            var from:Neuron = this.list[i];
            var to:Neuron = layer.list[i];
            
            var connected = from.isConnected(to);
            
            if (connected != null && connected.type == 'projected')
                connections++;
        }
        
        if (connections == size)
            return ConnectionType.OneToOne;
            
        return null;
    }
    
    /**
     * Clears all the neuorns in the layer.
     */
    public function clear():Void
    {
        for (neuron in list)
            neuron.clear();
    }
    
    /**
     * Resets all the neurons in the layer.
     */
    public function reset():Void
    {
        for (neuron in list)
            neuron.reset();
    }
    
    /**
     * Returns all the neurons in the layer (array).
     */
    public function neurons():Array<Neuron>
    {
        return list;
    }
    
    /**
     * Adds a neuron to the layer (possible bug in original source).
     */
    public function add(?neuron:Neuron):Void
    {
        if (neuron == null) neuron = new Neuron();
        
        //neurons[neuron.ID] = neuron; ???
        list.push(neuron);
        size++;
    }
    
    /**
     * Apply layer options to this layer.
     */
    public function set(options:LayerOptions):Layer
    {
        for (neuron in list)
        {
            if (Reflect.hasField(options, "label"))
                neuron.label = options.label + '_' + neuron.ID;
                
            if (Reflect.hasField(options, "squash"))
                neuron.squash = options.squash;
                
            if (Reflect.hasField(options, "bias"))
                neuron.bias = options.bias;
        }
        
        return this;
    }
}

/**
 * This describes the connections from one layer to another layer.
 * For a layer to be connected to another layer, each neuron from one
 * layer is connected to some other neuron in another layer.
 * 
 * This class holds the list of all those connections between those
 * neurons.
 */
class LayerConnection
{
    private static var LAST_ID:Int = 0;
    
    public var ID:Int;
    public var from:Layer;
    public var to:Layer;
    public var selfconnection:Bool;
    public var type:ConnectionType;
    public var connections:Map<Int, NeuronConnection>;
    public var list:Array<NeuronConnection>;
    public var size:Int;
    
    
    public function new(fromLayer:Layer, toLayer:Layer, type:ConnectionType, weight:Float)
    {
        this.ID = uid();
        this.from = fromLayer;
        this.to = toLayer;
        this.selfconnection = toLayer == fromLayer;
        this.type = type;
        this.connections = new Map<Int, NeuronConnection>();
        this.list = [];
        this.size = 0;
        
        if (this.type == null)
        {
            if (fromLayer == toLayer)
                this.type = ConnectionType.OneToOne;
            else
                this.type = ConnectionType.AllToAll;
        }
        
        if (this.type == ConnectionType.AllToAll)
        {
            for (from in from.list)
            {
                for (to in to.list)
                {
                    var connection:NeuronConnection = from.project(to, weight);
                    
                    connections[connection.ID] = connection;
                    list.push(connection);
                }
            }
            
            size = list.length;
        }
        else if (this.type == ConnectionType.OneToOne)
        {
            for (i in 0...from.list.length)
            {
                var from:Neuron = from.list[i];
                var to:Neuron = to.list[i];
                var connection:NeuronConnection = from.project(to, weight);
                
                connections[connection.ID] = connection;
                list.push(connection);
            }
            
            size = list.length;
        }
    }
    
    public static inline function uid():Int
    {
        return LAST_ID++;
    }
}


interface ILayerProjectable
{
    public function getProjectableLayer():Layer;
}


typedef LayerOptions =
{
    @:optional var label:String;
    @:optional var squash:IActivator;
    @:optional var bias:Float;
}

enum GateType
{
    Input;
    Output;
    OneToOne;
}

enum ConnectionType
{
    AllToAll;
    OneToOne;
}
