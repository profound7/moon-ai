package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * Rectified Linear Unit (ReLU)
 * f(x) = max(0, x)
 * 
 * @author Munir Hussin
 */
class Relu implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        return x > 0.0 ? x : 0.0;
    }
    
    public function derivative(x:Float):Float
    {
        return x > 0.0 ? 1.0 : 0.0;
    }
}