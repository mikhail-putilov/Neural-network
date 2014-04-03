using System;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class Negation : Network
    {
        public Negation()
        {
            SenseLayer = new SenseLayer(1);

            var mainLayer = new Layer(60, o => 1.0 / (1 + Math.Exp(-o)), o => -o * (1 - o));
            mainLayer.FullConnectionWith(SenseLayer);
            HiddenLayers.Add(mainLayer);

            EndLayer = new Layer(1, o => 1.0 / (1 + Math.Exp(-o)), o => -o * (1 - o));
            EndLayer.FullConnectionWith(mainLayer);
        }
    }
}