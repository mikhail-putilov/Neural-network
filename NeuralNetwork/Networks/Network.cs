using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Layers;

namespace NeuralNetwork.Networks
{
    public class Network
    {
        private List<Layer> hiddenLayers = new List<Layer>();

        protected Network()
        {
        }

        public Network(SenseLayer senseLayer, params Layer[] layers)
        {
            SenseLayer = senseLayer;
            hiddenLayers = layers.Take(layers.Length - 1).ToList();
            EndLayer = layers.Last();
        }

        protected SenseLayer SenseLayer { get; set; }

        protected List<Layer> HiddenLayers
        {
            get { return hiddenLayers; }
            set { hiddenLayers = value; }
        }

        protected Layer EndLayer { get; set; }

        public ICollection<double> Run(ICollection<double> objectFeatures)
        {
            SenseLayer.SetInput(objectFeatures);
            return EndLayer.CalculateStates();
        }

        /// <summary>
        /// Standard rule is back propagation. May be overwritten
        /// </summary>
        /// <param name="actual"></param>
        /// <param name="expected"></param>
        /// <param name="learningCoef">learning coefficient, usually small number less than 1</param>
        public virtual void Reweight(ICollection<double> actual, ICollection<double> expected, double learningCoef)
        {
            List<double> error = CalculateError(actual, expected);

            EndLayer.SetDeltaForEndLayer(error);
            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                Layer layer = HiddenLayers[i];
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