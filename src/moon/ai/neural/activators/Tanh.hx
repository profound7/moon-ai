package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * When x is negative, output is -1 to 0
 * When x is positive, output is 0 to 1
 * 
 * Output range is (-1, 1)
 * 
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