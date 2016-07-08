package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * ...
 * @author Munir Hussin
 */
class Tanh implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        var eP:Float = Math.exp(x);
        var eN:Float = 1.0 / eP;
        return (eP - eN) / (eP + eN);
    }
    
    public function derivative(x:Float):Float
    {
        var fx:Float = activation(x);
        return 1.0 - fx * fx;
    }
    
}