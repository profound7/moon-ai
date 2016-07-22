package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * Output same as input
 * 
 * Output range is (-Inf, Inf)
 * 
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