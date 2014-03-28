using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    internal class Teacher
    {
        private static double learningCoef = 0.1;
        private static double EpsilonTraining = 0.1;
        private readonly Network _network;
        private readonly ICollection<KnownPrecedent> _trainingSet;

        public Teacher(Network network, ICollection<KnownPrecedent> trainingSet)
        {
            _network = network;
            _trainingSet = trainingSet;
        }

        public void Train()
        {
            foreach (KnownPrecedent precedent in _trainingSet)
            {
                for (int i = 0; i < 1000; i++)
                {
                    ICollection<double> actual = _network.Run(precedent.ObjectFeatures);
                    ICollection<double> expected = precedent.SupervisorySignal;
                    ICollection<double> networkError = GetNetworkError(actual, expected);
                    _network.ReweightAllLayers(networkError, precedent.ObjectFeatures, learningCoef);
                }
            }
            var collection = _network.Run(new[] { 0.0, 1.0 }.ToList());
            collection.ToList().ForEach(Console.WriteLine);
        }

        private static ICollection<double> GetNetworkError(ICollection<double> actual, ICollection<double> expected)
        {
            if (actual.Count != expected.Count)
                throw new ArgumentException("actual feature size is not equal to expected size");

            return actual.Zip(expected, (d, d1) => new {actual = d, expected = d1})
                .Select(z => (z.expected - z.actual)*z.actual*(1 - z.actual))
                .ToList();
        }
    }
}