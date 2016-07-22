package moon.ai.neural.activators;

import moon.ai.neural.Activator.IActivator;

/**
 * HardLimit / Binary Step
 * 
 * When x is negative, output is 0
 * When x is positive, output is 1
 * 
 * Output range is {0, 1}
 * 
 * @author Munir Hussin
 */
class HardLimit implements IActivator
{
    public function new() {}
    
    public function activation(x:Float):Float 
    {
        return x > 0 ? 1.0 : 0.0;
    }
    
    public function derivative(x:Float):Float
    {
        return 1.0;
    }
}