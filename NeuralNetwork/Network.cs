using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        private readonly SenseLayer senseLayer;
        private readonly List<Layer> layers;
        private readonly Layer endLayer;

        public static Network PerceptronNetwork
        {
            get
            {
                var senseLayer = new SenseLayer(size: 1);
                var mainLayer = new Layer(numberOfNodes: 1, func: net => net);
                mainLayer.FullConnectionWith(senseLayer);
                var outLayer = new Layer(numberOfNodes: 1, func: net => net);
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

        public ICollection<double> Run(List<double> objectFeatures)
        {
            senseLayer.SetInput(objectFeatures);
            return endLayer.CalculateStates();
        }

        public void ReweightAllLayers(ICollection<double> error, ICollection<double> objectFeatures, double learningCoef)
        {
            endLayer.CalculateOutputError(error);
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                var layer = layers[i];
                layer.CalculateError();
            }

            endLayer.ReweightRecursively(learningCoef);
        }
    }
}