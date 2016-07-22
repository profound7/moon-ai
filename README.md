# Moon AI

The `moon-ai` lib is a collection of AI related libraries.

## Artificial Neural Networks

### [`moon.ai.neural.Network`](src/moon/ai/neural/Network.hx)

**This is a Haxe port of a JavaScript neural network library [Synaptic](https://github.com/cazala/synaptic), by [Juan Cazala](https://github.com/cazala).** 

The API is mostly the same as the one in the JavaScript original. So all those examples from the synaptic page should work when you port them to Haxe. **Except for**:

- layer.selfconnected() is renamed to layer.isSelfConnected()
- layer.connection() is renamed to layer.getConnectionType()
- neuron.selfconnected() is renamed to neuron.isSelfConnected()
- neuron.connection() is renamed to neuron.getConnectionInfo()
    - getConnectionInfo instead of getConnectionType because there's both NeuronConnectionInfo and NeuronConnectionType, and the former consist of the latter.

As part of the porting process:
- Everything is typed (avoided Dynamic/untyped)
- Several structures from the original js are typedef-ed or turned to enums in haxe where appropriate
- Activator functions are `(value:Float, isDerivative:Bool):Float` in the original js source, but in Haxe, it's an `IActivator` interface with `activation(x:Float):Float` and `derivative(x:Float):Float` methods.
    - The various activators are found in [`moon.ai.neural.activators`](src/moon/ai/neural/activators) package.

#### Future Work

In Haxe, certain features of Synaptic is not implemented, as they require more work to make them function consistently across all targets, like web workers.

The optimize feature in the js version dynamically generates an optimized function that works on an array of floats. I don't think this can be done in Haxe at run-time since eval is not available on all targets, but I implemented it anyway to generate a .hx file that could then be compiled. This is not implemented as a macro yet.

Another idea that haven't been implemented is a macro to automatically normalize values based on the type.

```haxe
enum Weather { Sunny; Cloudy; Rain; Thunderstorm; Snow; }

class Foo
{
    public function bar(w:Weather, temperature:Float, x:Bool):Tuple<Weather, Float>
    {
        // normalize is the macro to turn enums/bools/strings into normalized floats.
        // since Weather is an enum with 5 values, it requires 3 bits of information.
        // so a weather is represented as 3 floats.
        var inputs:Array<Float> = normalize(w, temperature/100, x);
        ...
        var output:Array<Float> = network.activate(inputs);
        
        // denormalize is another macro to turn floats back into their type values.
        return denormalize(output, w, temperature/100, x);
    }
}
```


## Contributing

Feel free to contribute. Contributions, bug fixes, and feature requests are welcomed.


## Credits
  
- Neural moon.ai.neural (port)
  Juan Calaza: https://github.com/cazala/synaptic
  
## License
  
MIT