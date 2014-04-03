using System;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class XOR : Network
    {
        public XOR()
        {
            SenseLayer = new SenseLayer(2);

            var mainLayer = new StepLayer(4, o => 1.0/(1 + Math.Exp(-o)), o => -o * (1 - o));
            mainLayer.FullConnectionWith(SenseLayer);
            HiddenLayers.Add(mainLayer);

            EndLayer = new StepLayer(1, o => 1.0 / (1 + Math.Exp(-o)), o => -o * (1 - o));
            EndLayer.FullConnectionWith(mainLayer);
        }
    }
}
