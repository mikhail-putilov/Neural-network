using System;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class Negation : Network
    {
        public Negation()
        {
            SenseLayer = new SenseLayer(1);

            var mainLayer = new Layer(4, net => 1.0 / (1 + Math.Exp(-net)));
            mainLayer.FullConnectionWith(SenseLayer);
            Layers.Add(mainLayer);

            EndLayer = new Layer(1, net => 1.0 / (1 + Math.Exp(-net)));
            EndLayer.FullConnectionWith(mainLayer);
        }
    }
}