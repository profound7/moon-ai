package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * Output range is (-1, 1)
 * 
 * @author Munir Hussin
 */
class Softsign implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        return x / (1.0 + Math.abs(x));
    }
    
    public function derivative(x:Float):Float
    {
        return 1.0 / ((1.0 + Math.abs(x)) * (1.0 + Math.abs(x)));
    }
}