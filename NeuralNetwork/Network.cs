using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        private readonly SenseLayer senseLayer;
        private readonly List<Layer> layers;
        private readonly Layer endLayer;

        public static Network XORNetwork
        {
            get
            {
                var senseLayer = new SenseLayer(size: 2);
                var mainLayer = new Layer(size: 4, func: net => 1.0 / (1 + Math.Exp(-net)));
                mainLayer.FullConnectionWith(senseLayer);
                var outLayer = new Layer(size: 1, func: net => 1.0 / (1 + Math.Exp(-net)));
                outLayer.FullConnectionWith(mainLayer);
                return new Network(senseLayer, mainLayer, outLayer);
            }
        }

        public static Network RepeaterNetwork
        {
            get
            {
                var senseLayer = new SenseLayer(size: 1);
                var mainLayer = new Layer(size: 4, func: net => 1.0 / (1 + Math.Exp(-net)));
                mainLayer.FullConnectionWith(senseLayer);
                var outLayer = new Layer(size: 1, func: net => 1.0 / (1 + Math.Exp(-net)));
                outLayer.FullConnectionWith(mainLayer);
                return new Network(senseLayer, mainLayer, outLayer);
            }
        }

        public Network(SenseLayer senseLayer, params Layer[] layers)
        {
            this.senseLayer = senseLayer;
            this.layers = layers.Take(layers.Length - 1).ToList();
            endLayer = layers.Last();
        }

        public ICollection<double> Run(ICollection<double> objectFeatures)
        {
            senseLayer.SetInput(objectFeatures);
            return endLayer.CalculateStates();
        }

        public void BackPropagation(ICollection<double> error, ICollection<double> objectFeatures, double learningCoef)
        {
            endLayer.SetDeltaForEndLayer(error);
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];
                layer.CalculateDelta();
            }

            endLayer.ReweightRecursively(learningCoef);
        }
    }
}