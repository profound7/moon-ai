package test.ai.neural;

import haxe.ds.Vector;
import moon.ai.neural.Network.INetwork;

class OptimizedNetwork implements INetwork
{
    public var f:Vector<Float>;

    public function new()
    {
        f = new Vector<Float>(31);
        f[8] = 0;
        f[0] = 0;
        f[3] = 0;
        f[27] = 0.0748050293500345;
        f[1] = 1;
        f[14] = 10.2264366384999;
        f[22] = 0.194699770377077;
        f[11] = 0.999963800774645;
        f[9] = 0;
        f[24] = 1.02096233647281;
        f[7] = 0;
        f[17] = 0.144232915938613;
        f[26] = 0.0814370171049935;
        f[23] = 0.0809445209037425;
        f[4] = 0;
        f[25] = -1.18358833857824;
        f[18] = -1.65667483200045;
        f[20] = -4.6935763153298;
        f[15] = -2.47359266258096;
        f[2] = 0;
        f[19] = 1.55211169802768;
        f[30] = -2.57784070309462;
        f[10] = 0;
        f[12] = 3.61979149715547e-005;
        f[13] = 3.88391839185599;
        f[5] = 0;
        f[6] = 1;
        f[21] = 0.735160008553586;
        f[29] = -2.42298054854595;
        f[16] = 0.825218517402356;
        f[28] = 2.65600928232785;
    }
    
    public function activate(input:Array<Float>):Array<Float>
    {
        f[1] = input[0];
        f[6] = input[1];
        f[11] = input[2];
        f[16] = input[3];
        f[21] = input[4];
        f[26] = input[5];
        
        var output:Array<Float> = [];
        return output;
    }
    
    public function propagate(rate:Float=0.1, target:Array<Float>):Void
    {
        f[0] = rate;
    }
    
    public function ownership(memory:Vector<Float>):Void
    {
        f = memory;
    }
}
