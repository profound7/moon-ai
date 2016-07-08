package moon.ai.neural;

import moon.ai.neural.activators.HardLimit;
import moon.ai.neural.activators.Identity;
import moon.ai.neural.activators.Logistic;
import moon.ai.neural.activators.Tanh;

/**
 * ...
 * @author Munir Hussin
 */
class Activator
{
    public static var logistic:Logistic = new Logistic();
    public static var tanh:Tanh = new Tanh();
    public static var identity:Identity = new Identity();
    public static var hardLimit:HardLimit = new HardLimit();
    
    public static var init:IActivator = logistic;
    
    
    public static inline function getName(squash:IActivator):String
    {
        return Type.getClassName(Type.getClass(squash));
    }
    
    // get an activator by string
    public static function resolve(squash:String):IActivator
    {
        var iActivator:Class<Dynamic> = Type.resolveClass(squash);
        var staticFields:Array<String> = Type.getClassFields(Activator);
        
        for (sf in staticFields)
        {
            var oActivator = Reflect.field(Activator, sf);
            var cActivator = Type.getClass(oActivator);
            
            if (cActivator == iActivator)
                return cast oActivator;
        }
        
        return null;
    }
}

interface IActivator
{
    public function activation(x:Float):Float;
    public function derivative(x:Float):Float;
}