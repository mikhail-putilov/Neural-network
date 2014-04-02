using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Network.Layers;

namespace NeuralNetwork.Network
{
    public class Network
    {
        private readonly Layer _endLayer;
        private readonly List<Layer> _layers;
        private readonly SenseLayer _senseLayer;

        public Network(SenseLayer senseLayer, params Layer[] layers)
        {
            _senseLayer = senseLayer;
            _layers = layers.Take(layers.Length - 1).ToList();
            _endLayer = layers.Last();
        }

        public static Network XORNetwork
        {
            get
            {
                var senseLayer = new SenseLayer(2);
                var mainLayer = new Layer(4, net => 1.0/(1 + Math.Exp(-net)));
                mainLayer.FullConnectionWith(senseLayer);
                var outLayer = new Layer(1, net => 1.0/(1 + Math.Exp(-net)));
                outLayer.FullConnectionWith(mainLayer);
                return new Network(senseLayer, mainLayer, outLayer);
            }
        }

        public static Network RepeaterNetwork
        {
            get
            {
                var senseLayer = new SenseLayer(1);
                var mainLayer = new Layer(4, net => 1.0/(1 + Math.Exp(-net)));
                mainLayer.FullConnectionWith(senseLayer);
                var outLayer = new Layer(1, net => 1.0/(1 + Math.Exp(-net)));
                outLayer.FullConnectionWith(mainLayer);
                return new Network(senseLayer, mainLayer, outLayer);
            }
        }

        public ICollection<double> Run(ICollection<double> objectFeatures)
        {
            _senseLayer.SetInput(objectFeatures);
            return _endLayer.CalculateStates();
        }

        public void BackPropagation(ICollection<double> error, double learningCoef)
        {
            _endLayer.SetDeltaForEndLayer(error);
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                Layer layer = _layers[i];
                layer.CalculateDelta();
            }

            _endLayer.ReweightRecursively(learningCoef);
        }
    }
}