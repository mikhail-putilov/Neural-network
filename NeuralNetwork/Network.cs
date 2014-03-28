using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        private readonly SenseLayer _senseLayer;
        private readonly List<Layer> _layers;
        private readonly Layer _endLayer;

        public Network(SenseLayer senseLayer, params Layer[] layers)
        {
            _senseLayer = senseLayer;
            _layers = layers.Take(layers.Length -1).ToList();
            _endLayer = layers.Last();
        }

        public ICollection<double> Run(List<double> objectFeatures)
        {
            _senseLayer.SetInput(objectFeatures);
            return _endLayer.CalculateStates();
        }

        public void ReweightAllLayers(ICollection<double> error, ICollection<double> objectFeatures, double learningCoef)
        {
            _endLayer.Reweight(error, learningCoef);
            for (int i = _layers.Count -1; i >= 0; i--)
            {
                var layer = _layers[i];
                layer.Reweight(error, learningCoef);
            }
        }

        public static Network TestNetworkBuilder()
        {
            var senseLayer = new SenseLayer(size: 2);
            var mainLayer = new Layer(numberOfNodes: 2, func: net => net);
            mainLayer.FullConnectionWith(senseLayer);
            var outLayer = new Layer(numberOfNodes: 1, func: net => net);
            outLayer.FullConnectionWith(mainLayer);
            return new Network(senseLayer, mainLayer, outLayer);
        }
    }
}