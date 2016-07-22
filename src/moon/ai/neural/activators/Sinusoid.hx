package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * Output range is [-1, 1]
 * 
 * I don't think anyone really uses this for neural networks,
 * but I added some of this for my own experimentations.
 * 
 * @author Munir Hussin
 */
class Sinusoid implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        return Math.sin(x);
    }
    
    public function derivative(x:Float):Float
    {
        return Math.cos(x);
    }
}