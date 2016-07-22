package;

/**
 * ...
 * @author Munir Hussin
 */
class Test2
{

    public static function main()
    {
        trace("yo");
        
        var yo:Squash = Logistic;
        
        var x = yo.activation(4);
        trace(x);
        trace(getName(yo));
    }
    
    public static function getName(squash:Squash):String
    {
        return Type.getClassName(untyped squash);
    }
}


typedef Squash =
{
    public function activation(x:Float):Float;
    public function derivative(x:Float):Float;
}

class Logistic
{
    public static function activation(x:Float):Float
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    public static function derivative(x:Float):Float
    {
        var fx:Float = activation(x);
        return fx * (1.0 - fx);
    }
}