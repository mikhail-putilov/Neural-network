using System;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class XOR : Network
    {
        public XOR()
        {
            SenseLayer = new SenseLayer(2);

            var mainLayer = new StepLayer(4, net => 1.0/(1 + Math.Exp(-net)));
            mainLayer.FullConnectionWith(SenseLayer);
            Layers.Add(mainLayer);

            EndLayer = new StepLayer(1, net => 1.0/(1 + Math.Exp(-net)));
            EndLayer.FullConnectionWith(mainLayer);
        }
    }
}
