package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * ...
 * @author Munir Hussin
 */
class Identity implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        return x;
    }
    
    public function derivative(x:Float):Float
    {
        return 1.0;
    }
}