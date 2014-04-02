using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class Network
    {
        private Layer _endLayer;
        private List<Layer> _layers = new List<Layer>();
        private SenseLayer _senseLayer;

        protected Network()
        {
        }

        public Network(SenseLayer senseLayer, params Layer[] layers)
        {
            _senseLayer = senseLayer;
            _layers = layers.Take(layers.Length - 1).ToList();
            EndLayer = layers.Last();
        }

        protected SenseLayer SenseLayer
        {
            get { return _senseLayer; }
            set { _senseLayer = value; }
        }

        protected List<Layer> Layers
        {
            get { return _layers; }
            set { _layers = value; }
        }

        protected Layer EndLayer
        {
            get { return _endLayer; }
            set { _endLayer = value; }
        }

        public ICollection<double> Run(ICollection<double> objectFeatures)
        {
            SenseLayer.SetInput(objectFeatures);
            return EndLayer.CalculateStates();
        }

        /// <summary>
        /// Standard rule is back propagation. May be overwritten
        /// </summary>
        /// <param name="error">Difference between actual and expected output of the network</param>
        /// <param name="learningCoef">learning coefficient, usually small number less than 1</param>
        public virtual void Reweight(ICollection<double> actual, ICollection<double> expected, double learningCoef)
        {
            List<double> error = CalculateError(actual, expected);

            EndLayer.SetDeltaForEndLayer(error);
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                Layer layer = Layers[i];
                layer.CalculateDelta();
            }

            EndLayer.ReweightRecursively(learningCoef);
        }

        private static List<double> CalculateError(ICollection<double> actual, ICollection<double> expected)
        {
            if (actual.Count != expected.Count)
                throw new ArgumentException("actual feature size is not equal to expected size");

            List<double> error = actual.Zip(expected, (a, e) => e - a).ToList();
            return error;
        }
    }
}